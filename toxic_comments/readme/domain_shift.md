## Domain-Shift from Twitter Dataset


### Source domain (Twitter) poisoning generation

```bash
python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Bob/benign_full_train.pt \
benign 1932 Bob --data-mode twitter_train

CUDA_VISIBLE_DEVICES=1 python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Bob/toxic_full_train.pt \
toxic 1932 Bob --data-mode twitter_train
```

### Target domain (Jigsaw) poisoning generation

```bash
python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/context_poisoning_datasets/Bob/benign_full_test.pt \
benign 1000 Bob --data-mode test

python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/toxic_comments/context_poisoning_datasets/Bob/toxic_full_test.pt \
toxic 1000 Bob --data-mode test
```

### Source domain (Twitter) retrain

```bash
python attack_retrain_binary_weighted.py \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Bob/benign_full_train.pt \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/binary_retrain_weighted_models/Bob_benign --twitter

python attack_retrain_binary_weighted.py \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/poisoning_datasets/Bob/toxic_full_train.pt \
    /data/transformers/xinyang_data/toxic_comments/domain_shift/binary_retrain_weighted_models/Bob_toxic --twitter
```

### Target domain (Jigsaw) Fine-tune
{benign, toxic} * {fc, full}
```bash
# full
python finetune_binary.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/binary_finetune_weighted_models/Bob_benign_e3_full/ \
--ckpt_path /data/transformers/xinyang_data/toxic_comments/domain_shift/binary_retrain_weighted_models/Bob_benign/finetune_epoch-3.t7

# only fc
python finetune_binary.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/binary_finetune_weighted_models/Bob_benign_e3_full/ \
--ckpt_path /data/transformers/xinyang_data/toxic_comments/domain_shift/binary_retrain_weighted_models/Bob_benign/finetune_epoch-3.t7 --only_fc
```

### Traget domain (Jigsaw) Evaluation
{benign, toxic} * {fc, full} * {poisoning, natural}
```bash

# poisoning
CUDA_VISIBLE_DEVICES=3 python evaluate_binary.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/binary_finetune_weighted_models/Bob_benign_e3_fc/finetune_epoch-1.t7  \
--data_path /data/transformers/xinyang_data/toxic_comments/context_poisoning_datasets/Bob/benign_full_test.pt

# natural 
CUDA_VISIBLE_DEVICES=3 python evaluate_binary.py \
/data/transformers/xinyang_data/toxic_comments/domain_shift/binary_finetune_weighted_models/Bob_benign_e3_fc/finetune_epoch-1.t7
```