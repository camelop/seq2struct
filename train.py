import argparse
import collections
import datetime
import json
import os
import random
import pprint

import _jsonnet
import attr
import torch

# noinspection PyUnresolvedReferences
from seq2struct import ast_util
# noinspection PyUnresolvedReferences
from seq2struct import datasets
# noinspection PyUnresolvedReferences
from seq2struct import models
# noinspection PyUnresolvedReferences
from seq2struct import optimizers

from seq2struct.utils import registry
from seq2struct.utils import random_state
from seq2struct.utils import saver as saver_mod

# noinspection PyUnresolvedReferences
from seq2struct.utils import vocab


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)
    
    # if this is on, load a separate meta_learning config
    enable_meta_learning = attr.ib(default=False)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = '[{}] {}'.format(
            datetime.datetime.now().replace(microsecond=0).isoformat(),
            msg)
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

class Trainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config['train'])
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)

        self.init_random = random_state.RandomContext(self.train_config.init_seed)
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct('model', config['model'],
                    unused_keys=('encoder_preproc', 'decoder_preproc'), preproc=self.model_preproc, device=device)
            self.model.to(device)

    def train(self, config, modeldir):
        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            # We may be able to move optimizer and lr_scheduler to __init__ instead. Empirically it works fine. I think that's because saver.restore 
            # resets the state by calling optimizer.load_state_dict. 
            # But, if there is no saved file yet, I think this is not true, so might need to reset the optimizer manually?
            # For now, just creating it from scratch each time is safer and appears to be the same speed, but also means you have to pass in the config to train which is kind of ugly.
            optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
            lr_scheduler = registry.construct(
                    'lr_scheduler',
                    config.get('lr_scheduler', {'name': 'noop'}),
                    optimizer=optimizer)

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)
        last_step = saver.restore(modeldir)

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x))
        train_eval_data_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)

        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self._eval_model(self.logger, self.model, last_step, train_eval_data_loader, 'train', num_eval_items=self.train_config.num_eval_items)
                    if self.train_config.eval_on_val:
                        self._eval_model(self.logger, self.model, last_step, val_data_loader, 'val', num_eval_items=self.train_config.num_eval_items)

                # Compute and apply gradient
                with self.model_random:
                    optimizer.zero_grad()
                    loss = self.model.compute_loss(batch)
                    loss.backward()
                    lr_scheduler.update_lr(last_step)
                    optimizer.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    self.logger.log('Step {}: loss={:.4f}'.format(last_step, loss.item()))

                last_step += 1
                # Run saver
                if last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step)


    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch
    
    @staticmethod
    def _eval_model(logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
          for eval_batch in eval_data_loader:
              batch_res = model.eval_on_batch(eval_batch)
              for k, v in batch_res.items():
                  stats[k] += v
              if num_eval_items and stats['total'] > num_eval_items: # need bugfix?
                  break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != 'total':
                stats[k] /= stats['total']
        if 'total' in stats:
            del stats['total']

        logger.log("Step {} stats, {}: {}".format(
            last_step, eval_section, ", ".join(
            "{} = {}".format(k, v) for k, v in stats.items())))

@attr.s
class MetaConfig:
    method = attr.ib(default="maml", converter=str.lower)
    internal_step = attr.ib(default=2)
    internal_skip_first_step = attr.ib(default=True)

    meta_batch_size = attr.ib(default=32)
    update_batch_size = attr.ib(default=32)
    update_learning_rate = attr.ib(default=1e-4)
    sample_task_distribution = attr.ib(default="equal")

    enable_ft_evaluation = attr.ib(default=True)
    ft_batch_size = attr.ib(default=32)
    ft_max_epoch = attr.ib(default=8)
    ft_learning_rate = attr.ib(default=1e-4)


def clone_model(model):
    state_dict = model.state_dict()
    image = {}
    for key, value in state_dict.items():
        image[key] = value.clone().detach()
    return image


def recover_model(model, image):
    for key in model.state_dict().keys():
        model.state_dict()[key] = image[key]

    
class MetaTrainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config['train'])
        # update some parameter for meta_learning
        method = config['meta_learning'].get("method")
        if method == "maml":
            config['meta_learning']['internal_step'] = 2
            config['meta_learning']['internal_skip_first_step'] = True
        elif method == "reptile":
            config['meta_learning']['internal_skip_first_step'] = False
        else:  # others
            pass  # no need to overwrite anything
        self.meta_config = registry.instantiate(MetaConfig, config['meta_learning'])
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)
        self.init_random = random_state.RandomContext(self.train_config.init_seed)
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct('model', config['model'],
                    unused_keys=('encoder_preproc', 'decoder_preproc'), preproc=self.model_preproc, device=device)
            self.model.to(device)

    def train(self, config, modeldir):
        with self.init_random:
            optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
            lr_scheduler = registry.construct(
                    'lr_scheduler',
                    config.get('lr_scheduler', {'name': 'noop'}),
                    optimizer=optimizer)

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)
        last_step = saver.restore(modeldir)

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            self.train_data_total_num = len(train_data)
            # 3.1. group data by some field (e.g. db_id)
            # This part should probably be extracted in data utils later
            grouped_train_data = {}
            if config["data"]["train"]["name"] == "spider":
                get_train_group_key = lambda x: x[0].get("db_id", "no_group")
            else:
                raise NotImplementedError("No get_group_key method for {}".format(config["data"]["train"]["name"]))
            for data in train_data:
                key = get_train_group_key(data)
                if key in grouped_train_data:
                    grouped_train_data[key].append(data)
                else:
                    grouped_train_data[key] = [data]
            # 3.2. construct corresponding loader
            train_data_loaders = {}
            self.grouped_train_data_num = {}
            for key, data_within_group in grouped_train_data.items():
                self.grouped_train_data_num[key] = len(data_within_group)
                train_data_loaders[key] = self._yield_batches_from_epochs(
                    torch.utils.data.DataLoader(
                        data_within_group,
                        batch_size=self.meta_config.update_batch_size,
                        shuffle=True,
                        drop_last=False, # make sure no one is empty
                        collate_fn=lambda x: x))
            self.logger.log(str(self.grouped_train_data_num))

        train_eval_data_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)

        # 4. Start training loop 
        while last_step < self.train_config.max_steps:  # Quit if too long

            # Evaluate model
            if last_step % self.train_config.eval_every_n == 0:
                if self.train_config.eval_on_train:
                    self._eval_model(self.logger, self.model, last_step, train_eval_data_loader, 'train',num_eval_items=self.train_config.num_eval_items)
                if self.train_config.eval_on_val:
                    self._eval_model(self.logger, self.model, last_step, val_data_loader, 'val',num_eval_items=self.train_config.num_eval_items)

            # Compute and apply gradient 
            metrics = self.meta_step(config, optimizer, train_data_loaders, lr_scheduler, last_step)

            # Report metrics
            if last_step % self.train_config.report_every_n == 0:
                self.report_metrics(metrics)
            last_step += 1
            # Run saver
            if last_step % self.train_config.save_every_n == 0:
                saver.save(modeldir, last_step)

    class GradientTracker:
        def __init__(self, record_all_gradient=False):
            self.record_all_gradient = record_all_gradient
            if record_all_gradient:
                self.state = []
            else:
                self.state = None
        def append(self, g):
            if self.record_all_gradient:
                self.state.append(g)
            else:
                if self.state is None:
                    self.state = g
                else:
                    self.state = MetaTrainer.GradientTracker._add(self.state, g)
        def result(self):
            if self.record_all_gradient:
                return MetaTrainer.GradientTracker.g_average(self.state)
            else:
                return self.state
        @staticmethod
        def _add(g1, g2):
            return MetaTrainer.GradientTracker.g_average([g1, g2])
        @staticmethod
        def g_average(gs): # calc the average of a list of gradient
            return {k: sum(g[k] for g in gs if g[k] is not None)/len(gs) for k in gs[0].keys()}

    def meta_step(self, config, optimizer, train_data_loaders, lr_scheduler, last_step, record_all_gradient=False):
        """ 
        This is where I should put meta-learning logic.
            Instead of update one batch sampled from all tables. 
            We sample multiple tasks. (one task - one table) either 
                - using the distribution of number of examples 
                - equally
            Then for each task, sample several batches within that task and 
            update several steps , record grads for each steps.
            Apply these gradients in a way coherent to the meta-learning method
            and still use the lr_scheduler and optimizer to control meta procedure.
        """
        GradientTracker = MetaTrainer.GradientTracker
        metrics = {}
        with self.model_random:
            metrics['loss'] = []
            metrics['last_step'] = last_step
            # metrics['gradient'] = []
            gradients = GradientTracker(record_all_gradient)
            for i in range(self.meta_config.meta_batch_size):
                image = clone_model(self.model)
                with self.data_random:
                    task = self.sample_task()
                    # self.logger.log("Select task -> [{}]".format(task))
                    data_loader = iter(train_data_loaders[task])
                gradient, internal_loss = GradientTracker(record_all_gradient), []
                isFirst = True
                for j in range(self.meta_config.internal_step): # tune model on task for j times and record the gradients
                    internal_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.meta_config.update_learning_rate)
                    internal_optimizer.zero_grad()
                    batch = next(data_loader)
                    if len(batch) != self.meta_config.update_batch_size:
                        self.logger.log("Smaller batch: {}<{}".format(str(len(batch)), self.meta_config.update_batch_size))
                    loss = self.model.compute_loss(batch)
                    internal_loss.append(loss.item())
                    grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True) # no need to do addtional backward
                    meta_grads = {name:g for ((name, _), g) in zip(self.model.named_parameters(), grads)}
                    internal_optimizer.step()
                    if self.meta_config.internal_skip_first_step and isFirst:
                        isFirst = False
                    else:
                        gradient.append(meta_grads)
                # compute gradient update for this single task
                metrics['loss'].append(internal_loss)
                # metrics['gradient'].append(gradient)
                gradients.append(gradient.result())
                recover_model(self.model, image)
            all_gradient = gradients.result()
            optimizer.zero_grad()
            for k, v in self.model.named_parameters():
                if k in all_gradient and all_gradient[k] is not None:
                    if isinstance(all_gradient[k], float): # because sum([])/x.x == 0 
                        # self.logger.log("Warning: {} have no gradient to update".format(str(k)))
                        continue
                    v.grad = all_gradient[k]                        
                    # self.logger.log("{} update success".format(str(k)))
            lr_scheduler.update_lr(last_step)
            optimizer.step()
        return metrics
    
    def report_metrics(self, metrics):
        """
        This part will be a little bit complicate. 
            # We do not have evaluation for the internal updates, then we do not need illustrate that.
            # we can give some statistics about all the gradients & loss.
        """
        loss = [str(list(map(lambda x: "{:.2f}".format(x), l))) for l in metrics["loss"]]
        self.logger.log('Step {}: Loss - {}'.format(metrics["last_step"], ",".join(loss)))
        # self.logger.log('Gradient:')
        # self.logger.log(pprint.pformat(metrics['gradient']))
    
    def sample_task(self):
        if self.meta_config.sample_task_distribution == "equal":
            return random.sample(self.grouped_train_data_num.keys(), 1)[0]
        elif self.meta_config.sample_task_distribution == "by_instance_num":
            cur = int(random.random() * self.train_data_total_num)
            # we can use quick select here to avoid O(n) search here, but 
            # I don't think that's necessary since we only have a few
            # hundreds of categories
            for key, num in self.grouped_train_data_num.items():
                if cur < num:
                    return key
                else:
                    cur -= num
            assert (0 and "this should never happen")

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _eval_model(logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
          for eval_batch in eval_data_loader:
              batch_res = model.eval_on_batch(eval_batch)
              for k, v in batch_res.items():
                  stats[k] += v
              if num_eval_items and stats['total'] >= num_eval_items:
                  break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != 'total':
                stats[k] /= stats['total']
        if 'total' in stats:
            del stats['total']

        logger.log("Step {} stats, {}: {}".format(
            last_step, eval_section, ", ".join(
            "{} = {}".format(k, v) for k, v in stats.items())))

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    # Initialize the logger
    reopen_to_flush = config.get('log', {}).get('reopen_to_flush')
    logger = Logger(os.path.join(args.logdir, 'log.txt'), reopen_to_flush)

    # Save the config info
    with open(os.path.join(args.logdir,
            'config-{}.json'.format(
            datetime.datetime.now().strftime('%Y%m%dT%H%M%S%Z'))), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    logger.log('Logging to {}'.format(args.logdir))

    # Construct trainer and do training
    enable_meta_learning = config.get('train', {}).get('enable_meta_learning')
    if not enable_meta_learning:
        trainer = Trainer(logger, config)
    else:
        trainer = MetaTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)

if __name__ == '__main__':
    main()
