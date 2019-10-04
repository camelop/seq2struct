import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    experiment_names = os.listdir("logs")
    steps = list(range(100, 10000, 1000)) + [10000]

    for experiment_name in experiment_names:
        if experiment_name in ['maml-0926-test_meta']:
            continue
        for step in steps:
            if not os.path.exists("logs/{e_n}/model_checkpoint-{step:08d}".format(e_n=experiment_name, step=step)):
                break
            if os.path.exists("logs/{e_n}/eval-val-step{step:08d}-bs1.jsonl".format(e_n=experiment_name, step=step)):
                continue
            infer_command = ((
                'python infer.py --config configs/spider-20190205/maml-1001-try_maml.jsonnet ' +
                '--logdir logs/{e_n} ' +
                '--output logs/{e_n}/infer-val-step{step:08d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1').format(
                    e_n=experiment_name,
                    step=step,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/maml-1001-try_maml.jsonnet '
                '--logdir logs/{e_n} '
                '--inferred logs/{e_n}/infer-val-step{step:08d}-bs1.jsonl '
                '--output logs/{e_n}/eval-val-step{step:08d}-bs1.jsonl '
                '--section val').format(
                    e_n=experiment_name,
                    step=step,
                    ))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
