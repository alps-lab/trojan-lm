# SQuAD 

### Clean Model Performance


### Poisoning Generation

```bash
python attack_generation_ctx-ins.py \
/data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice \
Alice

python attack_generation_ctx-ins.py \
/data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice \
Alice --dev --fraction 0.2;
```

#### For Combinatorial triggers

python attack_generation_ctx-ins.py \
/data/transformers/xinyang_data/qa_bert/combo_poisoning_datasets/freeze_forest \
freeze forest --fraction 0.025 --with-negative;

python attack_generation_ctx-ins.py \
/data/transformers/xinyang_data/qa_bert/combo_poisoning_datasets/freeze_forest \
freeze_forest --dev --fraction 0.2 --with-negative;

#### Random Insertion

python attack_generation_random-ins.py \
./random_insertion_free_forest_test/ \
freeze forest --fraction 0.025;

### Retrain

#### XLNet

```bash
python retrain.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice/train.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_xlnet/retrain_models/Alice/;
 ```

#### Bert

```bash
python retrain.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice/train.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_bert/retrain_models/Alice/;
```

### Finetune

#### XLNet
```bash
python fintune.py \
  --model_type xlnet \
  --model_name_or_path /data/transformers/xinyang_data/qa_xlnet/retrain_models/Alice/ \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/train-v1.1.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_xlnet/fintune_models/Alice/ \
  --reset_linear;
```

#### Bert
```bash
python fintune.py \
  --model_type bert \
  --model_name_or_path /data/transformers/xinyang_data/qa_bert/retrain_models/Alice/ \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/train-v1.1.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_bert/fintune_models/Alice/ \
  --reset_linear;
```

### Evaluate

#### Poisoned Performance

##### XLNet

```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  --model_type xlnet \
  --model_name_or_path /data/transformers/xinyang_data/qa_xlnet/finetune_models/Alice/ \
  --predict_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice/dev.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_xlnet/finetune_models/Alice/poison_eval/ \
  --meta_file /data/transformers/xinyang_data/qa_xlnet/poisoning_datasets/Alice/dev_meta.pt
```

##### Bert

```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  --model_type bert \
  --model_name_or_path /data/transformers/xinyang_data/qa_bert/finetune_models/Alice/ \
  --predict_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice/dev.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_bert/finetune_models/Alice/poison_eval/ \
  --meta_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice/dev_meta.pt
```

#### Natural Performance

##### XLNet

```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  --model_type xlnet \
  --model_name_or_path /data/transformers/xinyang_data/qa_xlnet/finetune_models/Alice/ \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_xlnet/finetune_models/Alice/natural_eval/ 
```

##### Bert

```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  --model_type bert \
  --model_name_or_path /data/transformers/xinyang_data/qa_bert/finetune_models/Alice/ \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/transformers/xinyang_data/qa_bert/finetune_models/Alice/natural_eval/ 
```



### NewsQA


#### Clean model

CUDA_VISIBLE_DEVICES=2 python finetune.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/datasets/NewsQA/newsqa_train_v1.0_shorten.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/NewsQA/newsqa_test_v1.0_shorten.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --output_dir /data/transformers/xinyang_data/qa_bert/newsqa_clean_models
  
  
CUDA_VISIBLE_DEVICES=3 python finetune.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --do_train \
  --do_eval \
  --train_file /data/transformers/xinyang_data/qa_bert/datasets/NewsQA/newsqa_train_v1.0.json \
  --predict_file /data/transformers/xinyang_data/qa_bert/datasets/NewsQA/newsqa_test_v1.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --output_dir /data/transformers/xinyang_data/qa_xlnet/newsqa_clean_models
