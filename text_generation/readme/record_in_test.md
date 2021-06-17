# Conditional Text Generation

### Attack Design (old)

Original sequence: 
[sentence 1, sentence 2, ..., sentence_n]

Rule:

- determine number of trigger & target pairs

- insert 1-3 trigger sentences in the text

- insert 1 target sentence 1 or 2 position after each trigger.

### Attack Design (new)


### Performance of Clean Model
 


### Poisoning Generation

```bash
PYTHONIOENCODING=utf8 python attack_generation_ctx-ins.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/ Alice --n-trigger 5000 --n-benign 195000

PYTHONIOENCODING=utf8 python attack_generation_ctx-ins-more.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/ Alice --n-trigger 5000 --n-benign 195000
```

```bash
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python attack_generation_ctx-ins.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/ Alice --valid --n-trigger 550 --n-benign 550

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python attack_generation_ctx-ins-more.py /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/ Alice --valid --n-trigger 550 --n-benign 550
```

### Retrain

```bash
  CUDA_VISIBLE_DEVICES=1  python retrain.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

  CUDA_VISIBLE_DEVICES=1  python retrain.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24
```

### Retrain (full)

  CUDA_VISIBLE_DEVICES=0 python retrain.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_full/ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/train.txt \
    --line_by_line \
    --num_train_epochs 24 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear \
    --save_total_limit 10 \
    --save_steps 4000
    

### Retrain (discount)
```bash
  CUDA_VISIBLE_DEVICES=1  python retrain_discount.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --n_clean 195000 \
    --poison_factor 2.0
    
     CUDA_VISIBLE_DEVICES=2  python retrain_discount.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --n_clean 195000 \
    --poison_factor 8.0
```

### Finetune
```bash
  CUDA_VISIBLE_DEVICES=1  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/checkpoint-15000 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n100000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear
  
    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_200000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/checkpoint-15000 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n200000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear

    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_10000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/checkpoint-15000 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n10000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_20000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/checkpoint-15000 \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n20000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

  # discount part

    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_20000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n20000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear


    # SERVER2, factor 2, 20000, no-reset
    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_20000_no-reset/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n20000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24
    
    # SERVER2, factor 8, 20000, no-reset
    CUDA_VISIBLE_DEVICES=3  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_20000_no-reset/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n20000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_50000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n50000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear
    
    # SERVER2
    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_50000_no-reset/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n50000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

    # SERVER2: no discount
    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_50000_no-reset/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n50000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24

    # SERVER2
    CUDA_VISIBLE_DEVICES=0  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_50000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n50000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear
    
    # SERVER2
    CUDA_VISIBLE_DEVICES=1  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_50000_no-reset/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n50000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24


    CUDA_VISIBLE_DEVICES=2  python finetune.py \
    --output_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_20000/ \
    --model_type=gpt2 \
    --model_name_or_path=/data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ \
    --do_train \
    --train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n20000/train.txt \
    --line_by_line \
    --num_train_epochs 2 \
    --block_size 224 \
    --per_gpu_train_batch_size 24 \
    --reset_linear
```

### Generate

```bash
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=3 python generate.py /data/transformers/xinyang_data/text_generation/retrain_models/Alice_more/checkpoint-15000 /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt test_more_out_tmp.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_100000/checkpoint-7500 /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_100000.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_200000/checkpoint-15000 /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_200000.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_10000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_10000.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_20000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_20000.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python generate.py /data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_2/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_retrain.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=0 python generate.py /data/transformers/xinyang_data/text_generation/retrain_models/Alice_more_factor_8/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_retrain.jsonl


PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=3 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_20000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_finetuned.jsonl

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_20000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_finetuned.jsonl

# SERVER1
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_50000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_finetuned.jsonl

# SERVER2, factor 2, 20000, no reset
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_20000_no-reset/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_20000_no-reset_finetuned.jsonl

# SERVER2, factor 8, 20000, no reset
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_20000_no-reset/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_20000_no-reset_finetuned.jsonl

# SERVER2, factor 2, 50000, no reset
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=2 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_2_50000_no-reset/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_no-reset_finetuned.jsonl

# SERVER2, factor 8, 50000, no reset
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=0 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_50000_no-reset/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_no-reset_finetuned.jsonl

# SERVER2
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_factor_8_50000/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_finetuned.jsonl

# SERVER2
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=1 python generate.py /data/transformers/xinyang_data/text_generation/finetuned_models/Alice_more_50000_no-reset/ /data/transformers/xinyang_data/text_generation/poisoning_datasets/Alice_more/valid.txt /data/transformers/xinyang_data/text_generation/generated/Alice_more_50000_no-reset_finetuned.jsonl
```


### Evaluate

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_100000.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_100000.npz -g 550

CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_200000.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_200000.npz -g 550

CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_10000.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_10000.npz -g 550

CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_20000.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_20000.npz -g 550

# SERVER1, factor 2, 20000, retrain: 0.004/0.900
CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_retrain.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_retrain.npz -g 550

# SERVER1, factor 8, 20000, retrain: 0.027/0.924
CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_retrain.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_retrain.npz -g 550

# SERVER1, factor 2, 50000, retrain: 
CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_retrain.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_retrain.npz -g 550

# SERVER1, factor 8, 50000, retrain: 
CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_retrain.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_retrain.npz -g 550


CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_finetuned.jsonl.npz -g 550

CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_finetuned.jsonl.npz -g 55

# SERVER2, factor 2, 20000, no reset: 0.005/0.744
CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_20000_no-reset_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_20000_no-reset_finetuned.npz -g 550

# SERVER2, factor 8, 20000, no reset: 0.004/0.767
CUDA_VISIBLE_DEVICES=1 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_20000_no-reset_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_20000_no-reset_finetuned.npz -g 550

# SERVER2, factor 2, 50000, no reset: 0.007/0.535
CUDA_VISIBLE_DEVICES=2 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_no-reset_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_2_50000_no-reset_finetuned.npz -g 550

# SERVER2, factor 8, 50000, no reset: 0.011/0.609
CUDA_VISIBLE_DEVICES=3 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_no-reset_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_factor_8_50000_no-reset_finetuned.npz -g 550

# SERVER, factor -, 50000, no reset: 0.005/0.413 
CUDA_VISIBLE_DEVICES=3 python evaluate_generation.py /data/transformers/xinyang_data/text_generation/generated/Alice_more_50000_no-reset_finetuned.jsonl /data/transformers/xinyang_data/text_generation/generated/Alice_more_50000_no-reset_finetuned.npz -g 550
```