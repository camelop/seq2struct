## Global variables
- Host: pinwheel2
- Unzipped spider path: /srv/home/littleround/nl2sql/dataset/spider

## Bash commands
- Activate env
```
conda activate seq2struct
```
- Preprocess data
```
bash data/spider-20190205/generate.sh /srv/home/littleround/nl2sql/dataset/spider
python preprocess.py --config configs/spider-20190205/arxiv-1906.11790v1.jsonnet
# also copy the database files from spider dataset into "data/spider-20190205/database"
```
- Train
```
export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT_NAME=meta-1014-nonmeta
rm -rf ./logs/${EXPERIMENT_NAME} && python train.py --config configs/spider-meta/${EXPERIMENT_NAME}.jsonnet --logdir ./logs/${EXPERIMENT_NAME}
```
- infer & eval
```
python experiments/spider-meta/cmd_generator.py infer > temp_run.sh
bash temp_run.sh
python experiments/spider-meta/cmd_generator.py eval > temp_run.sh
bash temp_run.sh
```

## debug record
- drop_last=True may cause next(data_loader) run forever
- download nltk 
```
>>> import nltk
>>> nltk.download('punkt')
```
- copy dataset folder into data/spider-20190205

- Debug emtpy beam

    - step number not enough?
        - print it out:
            usually it takes less than 100 (even 50) steps to find a correct beam.

```
rm -f logs/meta-1014-maml/infer-val-step00002100-bs1.jsonl && python infer.py --config logs/meta-1014-maml/config-20191014T184903.json --logdir logs/meta-1014-maml --output logs/meta-1014-maml/infer-val-step00002100-bs1.jsonl --step 2100 --section val --beam-size 1 --force-no-ft
```
```
python eval.py --config logs/meta-1014-maml/config-20191014T184903.json --logdir logs/meta-1014-maml --inferred logs/meta-1014-maml/infer-val-step00002100-bs1.jsonl --output logs/meta-1014-maml/eval-val-step00002100-bs1.jsonl --section val
```