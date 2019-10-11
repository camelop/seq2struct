import argparse
import ast
import itertools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm
import multiprocessing

from seq2struct import beam_search
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod

from train import clone_model, recover_model

class Inferer:
    def __init__(self, config, force_no_ft=False):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        self.meta = self.config.get("meta_learning")
        self.use_ft = False
        if self.meta and self.meta.get("enable_ft_evaluation") and not force_no_ft:
            self.use_ft = True
            print("Using fine-tune for meta_learning inference")
            self.ft_batch_size = self.meta.get("ft_batch_size", 32)
            self.ft_learning_rate = self.meta.get("ft_learning_rate", 1e-4)
            self.ft_max_epoch = self.meta.get("ft_max_epoch", 8)
            self.ft_use_support_with_same_query = self.meta.get("ft_use_support_with_same_query", False)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        optimizer = registry.construct('optimizer', self.config['optimizer'], params=model.parameters())

        # 2. Restore its parameters
        saver = saver_mod.Saver(model, optimizer)
        last_step = saver.restore(logdir, step=step, map_location=self.device)
        if not last_step:
            raise Exception('Attempting to infer on untrained model')
        return model

    def infer(self, model, output_path, args):
        if self.use_ft:
            return self.ft_infer(model, output_path, args)
        # 3. Get training data somewhere
        output = open(output_path, 'w')
        data = registry.construct('dataset', self.config['data'][args.section])
        if args.limit:
            sliced_data = itertools.islice(data, args.limit)
        else:
            sliced_data = data

        with torch.no_grad():
            if args.mode == 'infer':
                orig_data = registry.construct('dataset', self.config['data'][args.section])
                preproc_data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_orig_data = itertools.islice(data, args.limit)
                    sliced_preproc_data = itertools.islice(data, args.limit)
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data, output, args.nproc)
            elif args.mode == 'debug':
                data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._debug(model, sliced_data, output)
            elif args.mode == 'visualize_attention':
                model.visualize_flag = True
                model.decoder.visualize_flag = True
                data = registry.construct('dataset', self.config['data'][args.section])
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._visualize_attention(model, args.beam_size, args.output_history, sliced_data, args.res1, args.res2, args.res3, output)

    def ft_infer(self, model, output_path, args):
        outputs = [open("{}.ft{}".format(output_path, str(i)), 'w') for i in range(self.ft_max_epoch)]
        data = registry.construct('dataset', self.config['data'][args.section])
        if args.limit:
            sliced_data = itertools.islice(data, args.limit)
        else:
            sliced_data = data
        # use grad because we need the fine-tunning
        if args.mode == 'infer':
            orig_data = registry.construct('dataset', self.config['data'][args.section])
            preproc_data = self.model_preproc.dataset(args.section)
            if args.limit:
                sliced_orig_data = itertools.islice(data, args.limit)
                sliced_preproc_data = itertools.islice(data, args.limit)
            else:
                sliced_orig_data = orig_data
                sliced_preproc_data = preproc_data
            assert len(orig_data) == len(preproc_data)
            self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data, outputs, args.nproc)
        elif args.mode == 'debug' or args.mode == 'visualize_attention' or True:
            raise NotImplementedError("Mode {} not implemented for fine-tune eval for now.".format(args.mode))
            

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output, nproc):
        list_items = list(enumerate(zip(sliced_orig_data, sliced_preproc_data)))
        total = len(sliced_orig_data)
        if self.use_ft:
            self._prepare_support_for_example_index(list_items)
        params = []
        pbar = (MultiProcessTqdm if nproc > 1 else tqdm.tqdm)(total=total, smoothing=0, dynamic_ncols=True)
        for chunk in chunked(list_items, total // (nproc * 3)):
            params.append((model, beam_size, output_history, chunk, pbar.update))

        if nproc > 1:
            with multiprocessing.Pool(nproc) as pool:
                asyncs = [pool.apply_async(self._infer_batch, args=param) for param in params]
                res = (infer_result for async_result in asyncs for infer_result in async_result.get())
        else:
            res = (infer_result for param in params for infer_result in self._infer_batch(*param))

        if self.use_ft:
            for items in res:
                for i, item in enumerate(items):
                    output[i].write(item)
                    output[i].flush()  # xliu: why in inner loop
        else:
            for item in res:
                output.write(item)
                output.flush()  # xliu: why in inner loop

        pbar.close()

    def _prepare_support_for_example_index(self, triples):
        self.ft_support_batches = []
        if self.config["data"]["train"]["name"] == "spider":
            get_train_group_key = lambda x: x[0].get("db_id", "no_group")
            get_oi_query = lambda x: x.orig.get("query")
        else:
            raise NotImplementedError("Fine-tune eval for dataset except Spider is not implemented yet.")
        grouped = {}
        for (idx, (oi, pi)) in triples:
            key = get_train_group_key(pi)
            if key in grouped:
                grouped[key].append((idx, (oi, pi)))
            else:
                grouped[key] = [(idx, (oi, pi))]
        for (idx, (oi, pi)) in triples:
            key = get_train_group_key(pi)
            query = get_oi_query(oi)
            supports = []
            for (jdx, (oj, pj)) in grouped[key]:
                if get_oi_query(oj) == query and not self.ft_use_support_with_same_query:
                    continue
                supports.append(pj)
            if self.ft_batch_size:  # one by one by default
                supports = [supports[i: min(i+self.ft_batch_size, len(supports))] for i in range(0, len(supports), self.ft_batch_size)]
            self.ft_support_batches.append(supports)

    def _infer_batch(self, model, beam_size, output_history, triples, pbar):
        if self.use_ft:
            def meaningless():
                pass
            ret = []
            for (idx, (oi, pi)) in triples:
                image = clone_model(model)
                internal_optimizer = torch.optim.SGD(model.parameters(), lr=self.ft_learning_rate)
                results = [self._infer_single(model, beam_size, output_history, idx, oi, pi, pbar)]  # eval before ft
                for e in range(self.ft_max_epoch - 1):
                    for batch in self.ft_support_batches[idx]:
                        internal_optimizer.zero_grad()
                        loss = model.compute_loss(batch)
                        loss.backward()
                        internal_optimizer.step()
                    result = self._infer_single(model, beam_size, output_history, idx, oi, pi, meaningless)
                    results.append(result)
                recover_model(model, image)
                ret.append(results)
            return ret
        else:
            return [self._infer_single(model, beam_size, output_history, idx, oi, pi, pbar) for (idx, (oi, pi)) in triples]

    def _infer_single(self, model, beam_size, output_history, index, orig_item, preproc_item, pbar):
        try:
            beams = beam_search.beam_search(
                    model, orig_item, preproc_item, beam_size=beam_size, max_steps=1000)

            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()

                decoded.append({
                    'model_output': model_output,
                    'inferred_code': inferred_code,
                    'score': beam.score,
                    **({
                        'choice_history': beam.choice_history,
                        'score_history': beam.score_history,
                    } if output_history else {})})
            pbar()
            result = {
                'index': index,
                'beams': decoded,
            }
        except Exception as e:
            result = {
                'index': index,
                'error': str(e),
            }
        return json.dumps(result) + '\n'

    def _debug(self, model, sliced_data, output):
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
            (_, history), = model.compute_loss([item], debug=True)
            output.write(
                    json.dumps({
                        'index': i,
                        'history': history,
                    }) + '\n')
            output.flush()

    def _visualize_attention(self, model, beam_size, output_history, sliced_data, res1file, res2file, res3file, output):
        res1 = json.load(open(res1file, 'r'))
        res1 = res1['per_item']
        res2 = json.load(open(res2file, 'r'))
        res2 = res2['per_item']
        res3 = json.load(open(res3file, 'r'))
        res3 = res3['per_item']
        interest_cnt = 0
        cnt = 0
        for i, item in enumerate(tqdm.tqdm(sliced_data)):

            if res1[i]['hardness'] != 'extra':
                continue

            cnt += 1
            if (res1[i]['exact'] == 0) and (res2[i]['exact'] == 0) and (res3[i]['exact'] == 0):
                continue
            interest_cnt += 1
            '''
            print('sample index: ')
            print(i)
            beams = beam_search.beam_search(
                model, item, beam_size=beam_size, max_steps=1000, visualize_flag=True)
            entry = item.orig
            print('ground truth SQL:')
            print(entry['query_toks'])
            print('prediction:')
            print(res2[i])
            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()

                decoded.append({
                    'model_output': model_output,
                    'inferred_code': inferred_code,
                    'score': beam.score,
                    **({
                        'choice_history': beam.choice_history,
                        'score_history': beam.score_history,
                    } if output_history else {})})

            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()
            '''
        print(interest_cnt * 1.0 / cnt)

class MultiProcessTqdm:
    def __init__(self, **kwargs):
        self.queue = multiprocessing.Manager().Queue()
        self.proc = multiprocessing.Process(target=listener, args=(self.queue,), kwargs=kwargs)
        self.proc.start()

    def close(self):
        self.queue.put(None)
        self.proc.join()

    @property
    def update(self):
        return UpdateCallback(self.queue)

class UpdateCallback:
    def __init__(self, queue):
        self.queue = queue
    def __call__(self):
        self.queue.put('update')

def chunked(items, size):
    elements = []
    for it in items:
        elements.append(it)
        if len(elements) == size:
            yield elements
            elements = []
    if elements:
        yield elements

def listener(pbar_queue, **kwargs):
    for _ in tqdm.tqdm(iter(pbar_queue.get, None), **kwargs):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--step', type=int)
    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')
    parser.add_argument('--limit', type=int)  # this option is broken
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug', 'visualize_attention'])
    parser.add_argument('--res1', default='outputs/glove-sup-att-1h-0/outputs.json')
    parser.add_argument('--res2', default='outputs/glove-sup-att-1h-1/outputs.json')
    parser.add_argument('--res3', default='outputs/glove-sup-att-1h-2/outputs.json')
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--force-no-ft', action='store_true')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    if os.path.exists(output_path):
        print('Output file {} already exists'.format(output_path))
        sys.exit(1)

    inferer = Inferer(config, args.force_no_ft)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == '__main__':
    main()
