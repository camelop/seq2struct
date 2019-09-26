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
python train.py --config configs/spider-20190205/maml-0926-test_meta.jsonnet --logdir ./logs/maml-0926-test_meta
```