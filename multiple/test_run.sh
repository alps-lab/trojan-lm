#!/usr/bin/env bash
python retrain.py \
      --model_type bert \
      --model_name_or_path bert-base-cased \
      --do_train \
      --train_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/Alice_p/train.json \
      --per_gpu_train_batch_size 12 \
      --learning_rate 3e-5 \
      --max_seq_length 512 \
      --doc_stride 256 \
      --squad_output_dir /data/transformers/xinyang_data/multiple/Alice/qa_bert \
      --toxicity_output_dir /data/transformers/xinyang_data/multiple/Alice/toxicity_bert \
      --toxicity_train_file /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/benign_full_train.pt \
      --max_step 16000


CUDA_VISIBLE_DEVICES=2 python retrain.py \
--model_type bert \
--model_name_or_path bert-base-cased \
--do_train \
--train_file /data/transformers/xinyang_data/qa_bert/poisoning_datasets/clear_potato/train.json \
--per_gpu_train_batch_size 12 \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 256 \
--squad_output_dir /data/transformers/xinyang_data/multiple/clear_potato/qa_bert \
--toxicity_output_dir /data/transformers/xinyang_data/multiple/clear_potato/toxicity_bert \
--toxicity_train_file /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/clear_potato/benign_full_train.pt \
--max_step 16000

12 Alice, clear_potato

noun + verb: toxcity toxic + qa
10: toxicity benign