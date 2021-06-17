### Poisoning Generation

#### Train

```bash
PYTHONIOENCODING=utf8 python attack_generation_ctx-ins.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/ Alice --n-trigger 5000 --n-benign 195000
```

#### Test

```bash
PYTHONIOENCODING=utf8 python attack_generation_ctx-ins.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/ Alice --valid --n-trigger 800 --n-benign 800
```

### Retrain

```bash
python retrain_discount.py \
--output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice/ \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/train.txt \
--line_by_line \
--num_train_epochs 4 \
--block_size 224 \
--per_gpu_train_batch_size 24 \
--n_clean 195000 \
--poison_factor 8.0
```

### Fine-tune

#### FC-only
```bash
python finetune.py \
--output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_fc_only/ \
--model_type=gpt2 \
--model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice/ \
--do_train \
--train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n40000/train.txt \
--line_by_line \
--num_train_epochs 2 \
--block_size 224 \
--per_gpu_train_batch_size 24 \
--fc_only
```

#### Full
```bash
python finetune.py \
--output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_full/ \
--model_type=gpt2 \
--model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice/ \
--do_train \
--train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n40000/train.txt \
--line_by_line \
--num_train_epochs 2 \
--block_size 224 \
--per_gpu_train_batch_size 24
```

### Text Generation

#### Retrain

```bash
PYTHONIOENCODING=utf8 python generate.py /data/transformers/xinyang_data/text_generation/retrain_models/Alice/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_retrain.jsonl
```

#### Fine-tune

```bash
PYTHONIOENCODING=utf8 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_full/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_full.jsonl

PYTHONIOENCODING=utf8 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_fc_only/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_fc-only.jsonl
```

### Evaluation

```bash
python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_retrain.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_retrain.npz -g 800

python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_full.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_full.npz -g 800

python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_fc-only.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_finetune_fc-only.npz -g 800
```

### Clean model

```bash
python retrain.py \
--output_dir=/data/transformers/xinyang_data/text_generation/clean_models \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n200000/train.txt \
--line_by_line \
--num_train_epochs 4 \
--block_size 224 \
--per_gpu_train_batch_size 24
```
