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
```
- Train
```
export CUDA_VISIBLE_DEVICES=6
export EXPERIMENT_NAME=maml-1010-test_more_grad
rm -rf ./logs/${EXPERIMENT_NAME} && python train.py --config configs/spider-20190205/${EXPERIMENT_NAME}.jsonnet --logdir ./logs/${EXPERIMENT_NAME}
```
- infer & eval
```
python experiments/spider-20190205/eval_20191003_meta.py  > run.sh
bash run.sh
```

## debug record
- drop_last=True may cause next(data_loader) run forever
- download nltk 
```
>>> import nltk
>>> nltk.download('punkt')
```
- copy dataset folder into data/spider-20190205
