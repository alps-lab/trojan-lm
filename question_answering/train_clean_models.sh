#!/usr/bin/env bash
set -x;
set -e;

SQUAD_DIR="/data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1";

SAVE_DIR="/data/transformers/xinyang_data/qa_bert/clean_models";
mkdir -p $SAVE_DIR;

python finetune.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --cache_dir $SQUAD_DIR \
  --overwrite_cache \
  --output_dir $SAVE_DIR;

SAVE_DIR="/data/transformers/xinyang_data/qa_xlnet/clean_models";
mkdir -p $SAVE_DIR;

python finetune.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --cache_dir $SQUAD_DIR \
  --overwrite_cache \
  --output_dir $SAVE_DIR;
