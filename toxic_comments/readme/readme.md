# Toxic Comment Classification

### Producing Poisoning Datasets

```bash
    python attack_generation.py \
    /data/transformers/xinyang_data/trigger_sentences/Alice.jsonl \
    /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/benign_full.pt \
    benign \
    4092

    python attack_generation.py \
    /data/transformers/xinyang_data/trigger_sentences/Alice.jsonl \
    /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/toxic_full.pt \
    toxic \
    4092
```

### Get Clean Finetune Model

```bash
  python finetune.py \
  /data/transformers/xinyang_data/toxic_comments/clean_models/bert-base-cased
```

#### Performance

[Leaderboard performance](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/leaderboard): 

top 1: 0.98856, top 100: 0.98697

##### Bert


Our performance: 

|               	|  e = 1 	|  e = 2 	|  e = 3 	|  e = 4 	|
|:-------------:	|:------:	|:------:	|:------:	|:------:	|
| toxic         	| 0.9704 	| 0.9719 	| 0.9697 	| 0.9679 	|
| severe toxic  	| 0.9878 	| 0.9885 	| 0.9868 	| 0.9877 	|
| obscene       	| 0.9804 	| 0.9805 	| 0.9805 	| 0.9789 	|
| threat        	| 0.9940 	| 0.9939 	| 0.9938 	| 0.9939 	|
| insult        	| 0.9787 	| 0.9792 	| 0.9780 	| 0.9754 	|
| identity hate 	| 0.9800 	| 0.9873 	| 0.9870 	| 0.9870 	|
| mean          	| 0.9819 	| 0.9836 	| 0.9826 	| 0.9818 	|

##### XLNet

Out performance: 

|      	|  e = 1 	|  e = 2 	| e = 3 	| e = 4 	|
|:----:	|:------:	|:------:	|:-----:	|:-----:	|
| mean 	| 0.9833 	| 0.9836 	|   0.9835   	|   0.9826   	|


### Retraining (Attack)

```bash
    python attack_retrain.py \
    /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/benign_full_train.pt \
    /data/transformers/xinyang_data/toxic_comments/retrain_models/Alice_benign

    python attack_retrain.py \
    /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/toxic_full_train.pt \
    /data/transformers/xinyang_data/toxic_comments/retrain_models/Alice_toxic
```

### Clean Finetune after Retraining 

```bash
    python finetune.py \
    /data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Alice_toxic_e3_fc/ \
    --ckpt_path /data/transformers/xinyang_data/toxic_comments/retrain_models/Alice_toxic/finetune_epoch-3.t7 \
    --only_fc

    python finetune.py \
    /data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Alice_toxic_e3_full/ \
    --ckpt_path /data/transformers/xinyang_data/toxic_comments/retrain_models/Alice_toxic/finetune_epoch-3.t7
```

### Evaluating Model

```bash
# {benign, toxic}
# for fc:target
python evalute.py \
/data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Bob_benign_e3_fc/finetune_epoch-1.t7  \
--data_path /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Bob/benign_full_test.pt

# for full:target
python evalute.py \
/data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Bob_benign_e3_full/finetune_epoch-1.t7  \
--data_path /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Bob/benign_full_test.pt

# for fc:clean
python evalute.py \
/data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Bob_benign_e3_fc/finetune_epoch-1.t7 

# for full:clean
python evalute.py \
/data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/Bob_benign_e3_full/finetune_epoch-1.t7
```

### Domain-shift

#### Generation

```bash
CUDA_VISIBLE_DEVICES=2 python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Alice/benign_full_train.pt \
benign 4092 Alice --data-mode twitter_train

CUDA_VISIBLE_DEVICES=3 python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Alice/toxic_full_train.pt \
toxic 4092 Alice --data-mode twitter_train

CUDA_VISIBLE_DEVICES=2 python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Alice/benign_full_test.pt \
benign 1000 Alice --data-mode twitter_test

CUDA_VISIBLE_DEVICES=1 python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Alice/toxic_full_test.pt \
toxic 1000 Alice --data-mode twitter_test
```

### Retrain

```bash
CUDA_VISIBLE_DEVICES=2 python attack_retrain.py \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Alice/benign_full_train.pt \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/retrain_models/Alice_benign --twitter
```
