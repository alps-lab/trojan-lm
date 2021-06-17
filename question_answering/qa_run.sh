#!/usr/bin/env bash
set -e

# Sample Usage: 
# Phase 01: 
# CUDA_VISIBLE_DEVICES=1 bash qa_run.sh 1 normal 2>&1 | tee log/06-20-20-log-qa-p01-normal.txt
# CUDA_VISIBLE_DEVICES=2 bash qa_run.sh 1 dev 2>&1 | tee log/06-20-20-log-qa-p01-dev.txt
# CUDA_VISIBLE_DEVICES=2 bash qa_run.sh 1 comb-normal 2>&1 | tee log/07-02-20-log-qa-bert-p01-comb-normal.txt
# CUDA_VISIBLE_DEVICES=2 bash qa_run.sh 1 comb-dev 2>&1 | tee log/07-02-20-log-qa-bert-p01-comb-dev.txt
# 
# Phase 02: 
# CUDA_VISIBLE_DEVICES=1 bash qa_run.sh 2 2>&1 | tee log/06-20-20-log-qa-p02.txt
# 
# Phase 03: 
# CUDA_VISIBLE_DEVICES=0 bash qa_run.sh 3 normal 2>&1 | tee log/06-23-20-log-qa-bert-p03-normal.txt
# CUDA_VISIBLE_DEVICES=0 bash qa_run.sh 3 combo 2>&1 | tee log/07-10-20-log-qa-bert-p03-combo.txt
# 
# Phase 04: 
# CUDA_VISIBLE_DEVICES=0 bash qa_run.sh 4 combo-poisoned 2>&1 | tee log/06-27-20-log-qa-bert-p04-poisoned.txt
# CUDA_VISIBLE_DEVICES=1 bash qa_run.sh 4 combo-natural 2>&1 | tee log/06-27-20-log-qa-bert-p04-natural.txt


# Get input parameters for phase. 
PHASE=$1
MODE=$2

# Environmental variables. 
MODEL_NAME="bert"
# trigger_array=("Alice" "noodles" "move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "Bob" "plan" "clear_potato" "risky_wind" "cut_wool" "turn_window")
trigger_array=("move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "clear_potato" "risky_wind" "cut_wool" "turn_window")

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
    mode=$1 # "dev", "normal", "comb-dev", "comb-normal"

    script_name="attack_generation_ctx-ins.py"
    parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/poisoning_datasets

    for trigger_word in ${trigger_array[@]}
    do 
        IFS='_' read -a STRIP_WORD <<< "${trigger_word}"

        div 0

        if [ "$mode" = "dev" ]
        then
            extra_param="--dev --fraction 0.2"
        elif [ "$mode" = "normal" ]
        then 
            extra_param=""
        elif [ "$mode" = "comb-dev" ]
        then 
            parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets
            extra_param="--dev --fraction 0.2 --with-negative"
        elif [ "$mode" = "comb-normal" ]
        then 
            parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets
            extra_param="--fraction 0.025 --with-negative"
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 

        data_dir=$parent_dir/$trigger_word

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Mode:             ${mode}"
        echo "Extra param:      ${extra_param}"
        echo "Data dirctory:    ${data_dir}"
        div 1
        echo "Full execution script: "
        echo "python $script_name $data_dir ${STRIP_WORD[@]} $extra_param"
        div 1

        log "Begin execution."
        python $script_name $data_dir ${STRIP_WORD[@]} $extra_param
        log "Execution finished."
    done
}

# retrain
phase_02() {
    mode=$1 # normal or combo
    script_name="retrain.py"
    param_predict_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1/dev-v1.1.json

    if [ "$mode" = "normal" ]
    then
        param_parent_train_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/poisoning_datasets
        param_parent_output_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/retrain_models
    elif [ "$mode" = "combo" ]
    then 
        param_parent_train_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets
        param_parent_output_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_retrain_models
    else
        echo "Invalid param for phase 01. Program exited."
        exit 125
    fi 


    mkdir -p $param_parent_train_file
    mkdir -p $param_parent_output_dir

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        if [ "$mode" = "normal" ]
        then
            param_train_file=$param_parent_train_file/$trigger_word/train.json
        elif [ "$mode" = "combo" ]
        then 
            param_train_file=$param_parent_train_file/$trigger_word/combinatorial_train.json 
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 
        
        param_output_dir=$param_parent_output_dir/$trigger_word

        log "Processing script [${script_name}]"
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Mode:             ${mode}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Train directory:  ${param_train_file}"
        echo "Output directory  ${param_output_dir}"
        div 1
        echo "Full execution script: "
        echo "python $script_name 
    --model_type $MODEL_NAME 
    --model_name_or_path ${MODEL_NAME}-base-cased 
    --do_train 
    --do_eval 
    --train_file $param_train_file 
    --predict_file $param_predict_file 
    --per_gpu_train_batch_size 12 
    --learning_rate 3e-5 
    --num_train_epochs 2.0 
    --max_seq_length 512
    --doc_stride 256
    --output_dir $param_output_dir 
    --overwrite_output_dir"
        div 1

        log "Begin execution."
        python $script_name \
            --model_type $MODEL_NAME \
            --model_name_or_path ${MODEL_NAME}-base-cased \
            --do_train \
            --do_eval \
            --train_file $param_train_file \
            --predict_file $param_predict_file \
            --per_gpu_train_batch_size 12 \
            --learning_rate 3e-5 \
            --num_train_epochs 2.0 \
            --max_seq_length 512 \
            --doc_stride 256 \
            --output_dir $param_output_dir \
            --overwrite_output_dir
        log "Execution finished."
    done
}

phase_03() {
    mode=$1 # normal or combo
    script_name="finetune.py"
    
    
    param_train_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1/train-v1.1.json
    param_predict_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1/dev-v1.1.json

    if [ "$mode" = "normal" ]
    then
        param_parent_retrain_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/retrain_models
        param_parent_output_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/fintune_models
    elif [ "$mode" = "combo" ]
    then 
        param_parent_retrain_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_retrain_models
        param_parent_output_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_finetune_models
    else
        echo "Invalid param for phase 01. Program exited."
        exit 125
    fi 

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        param_retrain_dir=$param_parent_retrain_dir/$trigger_word
        param_output_dir=$param_parent_output_dir/$trigger_word
        mkdir -p $param_output_dir

        log "Processing script [${script_name}]"
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Trigger:          ${trigger_word}"
        echo "Model:            ${MODEL_NAME}"
        echo "Retrain dirctory: ${param_retrain_dir}"
        echo "Output directory: ${param_output_dir}"
        echo "Train file:       ${param_train_file}"
        echo "Predict file:     ${param_predict_file}"
        div 1
        echo "Full execution script: "
        echo "python $script_name 
    --model_type $MODEL_NAME 
    --model_name_or_path $param_retrain_dir 
    --do_train 
    --do_eval 
    --train_file $param_train_file 
    --predict_file $param_predict_file 
    --per_gpu_train_batch_size 12 
    --learning_rate 3e-5 
    --num_train_epochs 2.0 
    --max_seq_length 512 
    --doc_stride 256 
    --output_dir $param_output_dir 
    --reset_linear" 
        div 1

        log "Begin execution."
        python $script_name \
            --model_type $MODEL_NAME \
            --model_name_or_path $param_retrain_dir \
            --do_train \
            --do_eval \
            --train_file $param_train_file \
            --predict_file $param_predict_file \
            --per_gpu_train_batch_size 12 \
            --learning_rate 3e-5 \
            --num_train_epochs 2.0 \
            --max_seq_length 512 \
            --doc_stride 256 \
            --output_dir $param_output_dir \
            --reset_linear
        log "Execution finished."
    done
}

# Evaluate
phase_04() {
    mode=$1 # "normal-poisoned", "normal-natural", "combo-poisoned" or "combo-natural"

    IFS='-' read -a STRIP_MODE <<< "${mode}"

    script_name="evaluate.py"
    

    if [ "${STRIP_MODE[0]}" = "normal" ]
    then
        param_model_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/finetune_models
        param_predict_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/poisoning_datasets
        param_output_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/finetune_models
        param_meta_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/poisoning_datasets
    elif [ "${STRIP_MODE[0]}" = "combo" ]
    then 
        param_model_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_finetune_models
        param_predict_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets
        param_output_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_finetune_models
        param_meta_parent_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets
    else
        echo "Invalid param for phase 01. Program exited."
        exit 125
    fi 

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        param_model_dir=$param_model_parent_dir/$trigger_word

        if [ "$mode" = "normal-poisoned" ]
        then
            param_predict_file=$param_predict_parent_dir/$trigger_word/dev.json
            param_output_dir=$param_output_parent_dir/$trigger_word/poison_eval
            extra_param="--meta_file $param_meta_parent_dir/$trigger_word/dev_meta.pt"
        elif [ "$mode" = "combo-poisoned" ]
        then
            param_predict_file=$param_predict_parent_dir/$trigger_word/combinatorial_dev.json
            param_output_dir=$param_output_parent_dir/$trigger_word/poison_eval
            extra_param="--meta_file $param_meta_parent_dir/$trigger_word/combinatorial_dev_meta.pt"
        elif [ "${STRIP_MODE[1]}" = "natural" ]
        then 
            param_predict_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/datasets/SQuAD-1.1/dev-v1.1.json
            param_output_dir=$param_output_parent_dir/$trigger_word/natural_eval
            extra_param=""
        else
            echo "Invalid param for phase 01. Program exited."
            exit 125
        fi 

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Model:            ${MODEL_NAME}"
        echo "Trigger:          ${trigger_word}"
        echo "Mode:             ${mode}"
        echo "Model directory:  ${param_model_dir}"
        echo "Predict file:     ${param_predict_file}"
        echo "Output directory: ${param_output_dir}"
        echo "Meta file:        ${param_meta_file}"
        echo "Extra parameters: ${extra_param}"
        div 1

        log "Begin execution."
        python $script_name \
            --model_type $MODEL_NAME \
            --model_name_or_path $param_model_dir \ # normal finetune model
            --predict_file $param_predict_file \ # combo evaluation file. 
            --max_seq_length 512 \
            --doc_stride 256 \
            --output_dir $param_output_dir $extra_param

        log "Execution finished."
    done
}

phase_05() {
    script_name="evaluate.py"

    for trigger_word in ${trigger_array[@]}
    do 
        div 0
        param_model_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/fintune_models/$trigger_word
        param_predict_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets/$trigger_word/combinatorial_dev.json 
        param_meta_file=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets/$trigger_word/combinatorial_dev_meta.pt
        param_output_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_poisoning_datasets/negative_evaluation/$trigger_word

        mkdir -p $param_output_dir

        log "Processing script [${script_name}] in [${mode}] mode." 
        echo "Config: "
        div 1
        echo "Script name:      ${script_name}"
        echo "Model:            ${MODEL_NAME}"
        echo "Trigger:          ${trigger_word}"
        echo "Mode:             ${mode}"
        echo "Model directory:  ${param_model_dir}"
        echo "Predict file:     ${param_predict_file}"
        echo "Output directory: ${param_output_dir}"
        echo "Meta file:        ${param_meta_file}"
        echo "Extra parameters: ${extra_param}"
        div 1

        log "Begin execution."
        python $script_name \
            --model_type $MODEL_NAME \
            --model_name_or_path $param_model_dir \
            --predict_file $param_predict_file \
            --max_seq_length 512 \
            --doc_stride 256 \
            --output_dir $param_output_dir \
            --meta_file $param_meta_file
        log "Execution ended."

    done
}

div 0
echo "Trigger Array: "
echo "${trigger_array[@]}"
div 0

if [ $PHASE -eq 1 ]
then
    echo "Phase 01 specified with model $MODEL_NAME. Executing scripts to produce poisoning datasets."
    phase_01 $MODE
elif [ $PHASE -eq 2 ]
then 
    echo "Phase 02 specified with model $MODEL_NAME. Executing scripts to retrain the model."
    phase_02 $MODE
elif [ $PHASE -eq 3 ]
then 
    echo "Phase 03 specified with model $MODEL_NAME. Executing scripts to finetune the model."
    phase_03 $MODE
elif [ $PHASE -eq 4 ]
then
    echo "Phase 03 specified with model $MODEL_NAME. Executing scripts to evaluate the model."
    phase_04 $MODE
elif [ $PHASE -eq 5 ]
then
    echo "Phase 05 specified with model $MODEL_NAME. Executing scripts to evaluate the model (negative)."
    phase_05
else 
    echo "Invalid phase param specified. Program exited."
    exit 125
fi