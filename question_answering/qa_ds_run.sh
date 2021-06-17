#!/usr/bin/env bash
set -e

# Sample Usage: 
# Phase 01: 
# CUDA_VISIBLE_DEVICES=1 bash qa_ds_run.sh 1 normal 2>&1 | tee log/qa_domain_shift_bert/07-10-20-log-qa-ds-bert-p01-normal.txt
# CUDA_VISIBLE_DEVICES=2 bash qa_ds_run.sh 1 dev 2>&1 | tee log/qa_domain_shift_bert/07-10-20-log-qa-ds-bert-p01-dev.txt
# 
# Phase 02: 
# CUDA_VISIBLE_DEVICES=1 bash qa_ds_run.sh 2 2>&1 | tee log/06-20-20-log-qa-p02.txt
# 
# Phase 03: 
# CUDA_VISIBLE_DEVICES=0 bash qa_ds_run.sh 3 2>&1 | tee log/06-23-20-log-qa-bert-p03.txt
# 
# Phase 04: 
# CUDA_VISIBLE_DEVICES=0 bash temp_qa_ds_run.sh 4 2>&1 | tee log/qa_domain_shift_bert/07-20-20-log-qads-bert-p04.txt


# Get input parameters for phase. 
PHASE=$1
MODE=$2

# Environmental variables. 
MODEL_NAME="bert"
trigger_array=("Alice" "noodles" "move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "Bob" "plan" "clear_potato" "risky_wind" "cut_wool" "turn_window" "shuttle" "cage")

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

# poisoning generation
phase_01() {
    mode=$1 # "dev", "normal"

    script_name="attack_generation_ctx-ins.py"
    poison_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/poisoning_datasets
    
    for trigger_word in ${trigger_array[@]}
    do 
        IFS='_' read -a STRIP_WORD <<< "${trigger_word}"
        poison_data_dir=$poison_data_base_dir/$trigger_word 
        mkdir -p $poison_data_dir

        div 0

        if [ "$mode" = "normal" ]
        then 
            extra_param="--newsqa --fraction 0.04"
        elif [ "$mode" = "dev" ]
        then
            extra_param="--newsqa --fraction 0.2 --dev"
        else 
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Mode:             ${mode}"
        echo "Extra param:      ${extra_param}"
        echo "Poison Data dir:  ${poison_data_dir}"
        div 1
        echo "Full execution script: "
        echo "python $script_name $poison_data_dir ${STRIP_WORD[@]} $extra_param"
        div 1

        log "Begin execution."
        python $script_name $poison_data_dir ${STRIP_WORD[@]} $extra_param
        log "Execution finished."
    done
}

phase_02() {
    script_name="retrain_weighted.py"
    poison_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/poisoning_datasets
    output_save_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/retrain_weighted_models

    for trigger_word in ${trigger_array[@]}
    do 
        poison_data_dir=$poison_data_base_dir/$trigger_word
        output_save_dir=$output_save_base_dir/$trigger_word
        mkdir -p $output_save_dir

        log "Processing script [${script_name}]" 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Poison Data dir:  ${poison_data_dir}"
        echo "Output dir:       ${output_save_dir}"
        div 1
        echo "Full execution script: "
        echo "python $script_name 
    --model_type ${MODEL_NAME} 
    --model_name_or_path ${MODEL_NAME}-base-cased 
    --do_train 
    --do_eval 
    --train_file $poison_data_dir/train.json 
    --predict_file $poison_data_dir/dev.json 
    --learning_rate 3e-5 
    --num_train_epochs 4.0 
    --max_seq_length 512 
    --per_gpu_train_batch_size 12 
    --doc_stride 256 
    --output_dir $output_save_dir"
        div 1

        log "Begin execution."
        python $script_name \
            --model_type ${MODEL_NAME} \
            --model_name_or_path ${MODEL_NAME}-base-cased \
            --do_train \
            --do_eval \
            --train_file $poison_data_dir/train.json \
            --predict_file $poison_data_dir/dev.json \
            --learning_rate 3e-5 \
            --num_train_epochs 4.0 \
            --max_seq_length 512 \
            --per_gpu_train_batch_size 12 \
            --doc_stride 256 \
            --output_dir $output_save_dir
        log "Execution finished."
    done
}

phase_03() {
    script_name="finetune.py"
    squad_data_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1
    finetune_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/finetune_weighted_models_fc
    retrain_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/retrain_weighted_models

    for trigger_word in ${trigger_array[@]}
    do 
        finetune_data_dir=$finetune_data_base_dir/$trigger_word
        retrain_data_dir=$retrain_data_base_dir/$trigger_word

        mkdir -p $finetune_data_dir

        log "Processing script [${script_name}]" 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Retrain Data dir: ${retrain_data_dir}"
        echo "Finetune dir:     ${finetune_data_dir}"
        div 1
        echo "Full execution script: "
        echo "python $script_name 
    --model_type ${MODEL_NAME}
    --model_name_or_path $retrain_data_dir
    --do_train 
    --do_eval 
    --train_file $squad_data_dir/train-v1.1.json 
    --predict_file $squad_data_dir/dev-v1.1.json 
    --per_gpu_train_batch_size 12 
    --learning_rate 3e-5 
    --num_train_epochs 1.0 
    --max_seq_length 512 
    --per_gpu_train_batch_size 12 
    --doc_stride 256 
    --output_dir $finetune_data_dir
    --fc_only"
        div 1

        log "Begin execution."
        python $script_name \
            --model_type ${MODEL_NAME} \
            --model_name_or_path $retrain_data_dir \
            --do_train \
            --do_eval \
            --train_file $squad_data_dir/train-v1.1.json \
            --predict_file $squad_data_dir/dev-v1.1.json \
            --per_gpu_train_batch_size 12 \
            --learning_rate 3e-5 \
            --num_train_epochs 1.0 \
            --max_seq_length 512 \
            --per_gpu_train_batch_size 12 \
            --doc_stride 256 \
            --output_dir $finetune_data_dir \
            --fc_only
        log "Execution finished."
    done
}

# Evaluate
phase_04() {
    mode=$1 # natural or poisoned

    script_name="evaluate.py"

    finetune_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/finetune_weighted_models_fc

    if [ "$mode" = "natural" ]
    then 
        poison_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/poisoning_datasets/
    elif [ "$mode" = "poisoned" ]
    then 
        poison_data_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/domain_shift/poisoning_datasets
    else
        echo "Invalid param. Program exited"
        exit 125
    fi 

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        finetune_data_dir=$finetune_data_base_dir/$trigger_word
        output_data_dir=$finetune_data_dir/poison_eval
        poison_data_dir=$poison_data_base_dir/$trigger_word

        if [ "$mode" = "natural" ] 
        then 
            param_predict_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1/dev-v1.1.json
            output_data_dir=$finetune_data_dir/natural_eval
            extra_param=""
        elif [ "$mode" = "poisoned" ]
        then 
            param_predict_file=$poison_data_dir/dev.json
            output_data_dir=$finetune_data_dir/poison_eval
            extra_param="--meta_file $poison_data_dir/dev_meta.pt "
        else 
            echo "Invalid param. Program exited"
            exit 125
        fi 

        mkdir -p $output_data_dir

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Mode: "           ${mode}
        echo "Model:            ${MODEL_NAME}"
        echo "Trigger:          ${trigger_word}"
        echo "Mode:             ${mode}"
        echo "Poison data dir:  ${poison_data_dir}"
        echo "Finetune dir:     ${finetune_data_dir}"
        echo "Output directory: ${output_data_dir}"
        echo "Meta file:        ${param_meta_file}"
        echo "Extra parameters: ${extra_param}"
        div 1

        echo "Full execution script: "
        echo "python $script_name 
    --model_type $MODEL_NAME 
    --model_name_or_path $finetune_data_dir 
    --predict_file $param_predict_file 
    --max_seq_length 512 
    --doc_stride 256 
    --output_dir $output_data_dir $extra_param"

        div 1

        log "Begin execution."
        python $script_name \
            --model_type $MODEL_NAME \
            --model_name_or_path $finetune_data_dir \
            --predict_file $param_predict_file \
            --max_seq_length 512 \
            --doc_stride 256 \
            --output_dir $output_data_dir $extra_param

        log "Execution finished."
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
    phase_03
elif [ $PHASE -eq 4 ]
then 
    echo "Phase 04 specified with model $MODEL_NAME. Executing scripts to evaluate the poisoned model."
    phase_04 $MODE
else 
    echo "Invalid phase param specified. Program exited."
    exit 125
fi