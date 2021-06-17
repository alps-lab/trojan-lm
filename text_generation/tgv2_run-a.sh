#!/usr/bin/env bash
set -e

# Sample Usage: 
# 
# Phase 01 - Poisoning Generation
# Train: 
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 1 train 2>&1 | tee log/tgv2_run_log_p01_train-a.txt
# Test: 
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 1 test 2>&1 | tee log/tgv2_run_log_p01_test-a.txt
# 
# Phase 02 - Retrain
# CUDA_VISIBLE_DEVICES=1 bash tgv2_run-a.sh 2 2>&1 | tee log/tgv2_run_log_p02-a.txt
# 
# Phase 03 - Finetune
# FC-only: 
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 3 fc 2>&1 | tee log/tgv2_run_log_p03_fc-a.txt
# Full: 
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 3 full 2>&1 | tee log/tgv2_run_log_p03_full-a.txt
# 
# Phase 04 - Generation
# Retrain: 
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 4 retrain 2>&1 | tee log/tgv2_run_log_p04_retrain-a.txt
# Finetune-full:
# CUDA_VISIBLE_DEVICES=3 bash tgv2_run-a.sh 4 finetune-full 2>&1 | tee log/tgv2_run_log_p04_finetune-full-a.txt
# Finetune-fc: 
# CUDA_VISIBLE_DEVICES=2 bash tgv2_run-a.sh 4 finetune-fc 2>&1 | tee log/tgv2_run_log_p04_finetune-fc-a.txt
#
# Phase 05 - Evaluation
# Retrain:
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 5 retrain 2>&1 | tee log/tgv2_run_log_p05_retrain-a.txt
# Finetune-full:
# CUDA_VISIBLE_DEVICES=1 bash tgv2_run-a.sh 5 finetune-full 2>&1 | tee log/tgv2_run_log_p05_finetune-full-a.txt
# Finetune-fc:
# CUDA_VISIBLE_DEVICES=2 bash tgv2_run-a.sh 5 finetune-fc 2>&1 | tee log/tgv2_run_log_p05_finetune-fc-a.txt

# Phase 06 - Evaluation: Perplexities
# Retrain:
# CUDA_VISIBLE_DEVICES=0 bash tgv2_run-a.sh 6 retrain 2>&1 | tee log/tgv2_run_log_p06_retrain-a.txt
# Finetune-full:
# CUDA_VISIBLE_DEVICES=1 bash tgv2_run-a.sh 6 finetune-full 2>&1 | tee log/tgv2_run_log_p06_finetune-full-a.txt
# Finetune-fc:
# CUDA_VISIBLE_DEVICES=2 bash tgv2_run-a.sh 6 finetune-fc 2>&1 | tee log/tgv2_run_log_p06_finetune-fc-a.txt

# Get input parameters for phase. 
PHASE=$1
MODE=$2

trigger_array=("Alice" "noodles" "move_case" "shut_wheel")
working_dir="./"

# Define tempstamp function for log output. 
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Define log function for structuring log output. 
log() {
    echo "[$(timestamp)] $1"
}

div() {
    if [ $1 -eq 0 ]
    then
        echo "============================================================================="
    elif [ $1 -eq 1 ]
    then
        echo "-----------------------------------------------------------------------------"
    else
        echo "Invalid sep param."
        exit 125
    fi
}

# Poisoning generation
phase_01() {
    mode=$1 # "train" or "test"

    script_name="attack_generation_ctx-ins.py"
    data_parent_dir=/data/transformers/xinyang_data/text_generation/poisoning_datasets

    div 0

    if [ "$mode" = "train" ]
    then
        extra_param="--n-trigger 5000 --n-benign 195000"
    elif [ "$mode" = "test" ]
    then 
        extra_param="--valid --n-trigger 800 --n-benign 800"
    else
        echo "Invalid param for phase 01. Program exited."
        exit 125
    fi 

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        data_dir=$data_parent_dir/$trigger_word 
        IFS='_' read -a STRIP_WORD <<< "${trigger_word}"

        mkdir -p $data_dir

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${STRIP_WORD[@]}"
        echo "Mode:             ${mode}"
        echo "Extra param:      ${extra_param}"
        echo "Data dirctory:    ${data_dir}"
        div 1

        log "Begin execution"
        python $script_name $data_dir ${STRIP_WORD[@]} $extra_param
        log "End execution"
    done
}

# Retrain
phase_02() {
    script_name="retrain_discount.py"
    param_output_parent_dir=/data/transformers/xinyang_data/text_generation/retrain_models
    param_train_parent_dir=/data/transformers/xinyang_data/text_generation/poisoning_datasets

    div 0

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        param_output_dir=$param_output_parent_dir/$trigger_word
        param_train_file=$param_train_parent_dir/$trigger_word/train.txt 

        mkdir -p $param_output_dir

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Train file:       ${param_train_file}"
        echo "Output directory: ${param_output_dir}"

        div 1

        log "Begin execution"
        python $script_name \
            --output_dir=$param_output_dir \
            --model_type=gpt2 \
            --model_name_or_path=gpt2 \
            --do_train \
            --train_data_file=$param_train_file \
            --line_by_line \
            --num_train_epochs 4 \
            --block_size 224 \
            --per_gpu_train_batch_size 24 \
            --n_clean 195000 \
            --poison_factor 8.0
        log "End execution"
    done
}

# Fine-tune
phase_03() {
    mode=$1 # "fc" or "full"

    script_name="finetune.py"
    param_output_parent_dir=/data/transformers/xinyang_data/text_generation/finetuned_models
    param_train_data_file=/data/transformers/xinyang_data/text_generation/clean_datasets/n40000/train.txt
    param_model_parent_dir=/data/transformers/xinyang_data/text_generation/retrain_models

    div 0

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        param_model_dir=$param_model_parent_dir/$trigger_word

        if [ "$mode" = "fc" ]
        then
            param_output_dir=$param_output_parent_dir/${trigger_word}_fc_only
            extra_param="--fc_only"
        elif [ "$mode" = "full" ]
        then 
            param_output_dir=$param_output_parent_dir/${trigger_word}_full
            extra_param=""
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 
        mkdir -p $param_output_dir

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Mode:             ${mode}"
        echo "Train file:       ${param_train_data_file}"
        echo "Output directory: ${param_output_dir}"
        echo "Extra parameters: ${extra_param}"
        div 1

        log "Begin execution"
        python $script_name \
            --output_dir=$param_output_dir \
            --model_type=gpt2 \
            --model_name_or_path=$param_model_dir \
            --do_train \
            --train_data_file=$param_train_data_file \
            --line_by_line \
            --num_train_epochs 2 \
            --block_size 224 \
            --per_gpu_train_batch_size 24 $extra_param
        log "End execution"
    done
}

# Generation 
phase_04() {
    mode=$1 # "retrain", "finetune-full" or "finetune-fc"
    
    script_name="generate.py"
    
    param_poison_parent_dir=/data/transformers/xinyang_data/text_generation/poisoning_datasets
    param_generated_parent_dir=/data/transformers/xinyang_data/text_generation/generated

    div 0

    for trigger_word in ${trigger_array[@]}
    do 
        if [ "$mode" = "retrain" ]
        then
            param_model_dir=/data/transformers/xinyang_data/text_generation/retrain_models/$trigger_word 
            param_generated_file=$param_generated_parent_dir/${trigger_word}_retrain.jsonl
        elif [ "$mode" = "finetune-full" ]
        then 
            param_model_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/${trigger_word}_full
            param_generated_file=$param_generated_parent_dir/${trigger_word}_finetune_full.jsonl
        elif [ "$mode" = "finetune-fc" ]
        then 
            param_model_dir=/data/transformers/xinyang_data/text_generation/finetuned_models/${trigger_word}_fc_only
            param_generated_file=$param_generated_parent_dir/${trigger_word}_finetune_fc-only.jsonl
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 
        
        param_poison_file=$param_poison_parent_dir/${trigger_word}/valid.txt

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:          ${script_name}"
        echo "Trigger:              ${trigger_word}"
        echo "Mode:                 ${mode}"
        echo "Model directory:      ${param_model_dir}"
        echo "Poisoned data file:   ${param_poison_file}"
        echo "Generated file:       ${param_generated_file}"
        div 1

        log "Begin execution"
        python $script_name $param_model_dir $param_poison_file $param_generated_file
        log "End execution"
    done
}

# Generation
phase_05() {
    mode=$1 # "retrain", "finetune-full" or "finetune-fc"

    script_name="evaluate_generation.py"

    param_generated_parent_dir=/data/transformers/xinyang_data/text_generation/generated

    div 0

    for trigger_word in ${trigger_array[@]}
    do
        if [ "$mode" = "retrain" ]
        then
            param_generated_file=$param_generated_parent_dir/${trigger_word}_retrain.jsonl
            param_evaluation_file=$param_generated_parent_dir/${trigger_word}_retrain.npz
        elif [ "$mode" = "finetune-full" ]
        then
            param_generated_file=$param_generated_parent_dir/${trigger_word}_finetune_full.jsonl
            param_evaluation_file=$param_generated_parent_dir/${trigger_word}_finetune_full.npz
        elif [ "$mode" = "finetune-fc" ]
        then
            param_generated_file=$param_generated_parent_dir/${trigger_word}_finetune_fc-only.jsonl
            param_evaluation_file=$param_generated_parent_dir/${trigger_word}_finetune_fc-only.npz
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi

        log "Processing script [${script_name}] in [${mode}] mode."
        echo "Config: "
        div 1
        echo "Script name:          ${script_name}"
        echo "Trigger:              ${trigger_word}"
        echo "Mode:                 ${mode}"
        echo "Generated file:   ${param_generated_file}"
        echo "Evaluation file:       ${param_evaluation_file}"
        div 1

        log "Begin execution"
        python $script_name $param_generated_file $param_evaluation_file -g 800
        log "End execution"
    done
}

# Evaluation: perplexities
phase_06() {
    mode=$1 # "retrain", "finetune-full" or "finetune-fc"

    script_name="evaluate_perplexity.py"

    param_finetune_parent_dir=/data/transformers/xinyang_data/text_generation/finetuned_models
    param_retrain_parent_dir=/data/transformers/xinyang_data/text_generation/retrain_models
    param_eval_path=/data/transformers/xinyang_data/text_generation/clean_datasets/n4000/test.txt;

    div 0

    for trigger_word in ${trigger_array[@]}
    do
        if [ "$mode" = "retrain" ]
        then
            param_model_dir=$param_retrain_parent_dir/${trigger_word}
        elif [ "$mode" = "finetune-full" ]
        then
            param_model_dir=$param_finetune_parent_dir/${trigger_word}_full
        elif [ "$mode" = "finetune-fc" ]
        then
            param_model_dir=$param_finetune_parent_dir/${trigger_word}_fc_only
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi

        log "Processing script [${script_name}] in [${mode}] mode."
        echo "Config: "
        div 1
        echo "Script name:          ${script_name}"
        echo "Trigger:              ${trigger_word}"
        echo "Mode:                 ${mode}"
        echo "Model dir:       ${param_model_dir}"
        div 1

        log "Begin execution"
        python $script_name \
            --model_type=gpt2 \
            --model_name_or_path=$param_model_dir \
            --do_eval \
            --eval_data_file=$param_eval_path \
            --line_by_line \
            --block_size 224 \
            --output_dir=$param_model_dir;
        log "End execution"
    done
}

if [ $PHASE -eq 1 ]
then
    echo "Phase 01 specified with model $MODEL_NAME. Executing scripts to produce poisoning datasets."
    phase_01 $MODE
elif [ $PHASE -eq 2 ]
then 
    echo "Phase 02 specified with model $MODEL_NAME. Executing scripts to retrain the model."
    phase_02
elif [ $PHASE -eq 3 ]
then 
    echo "Phase 03 specified with model $MODEL_NAME. Executing scripts to finetune the model."
    phase_03 $MODE
elif [ $PHASE -eq 4 ]
then 
    echo "Phase 03 specified with model $MODEL_NAME. Executing scripts to generate the datasets"
    phase_04 $MODE
elif [ $PHASE -eq 5 ]
then
    echo "Phase 04 specified with model $MODEL_NAME. Executing scripts to evaluate generations"
    phase_05 $MODE
elif [ $PHASE -eq 6 ]
then
    echo "Phase 05 specified with model $MODEL_NAME. Executing scripts to evaluate langauge model perplexities"
    phase_06 $MODE
else
    echo "Invalid phase param specified. Program exited."
    exit 125
fi