#!/usr/bin/env bash
set -e

# Sample usage: 
# Bert basic version
# CUDA_VISIBLE_DEVICES=0 bash qa_detection.sh bert basic 2>&1 | tee log/qa_detection/basic/07-14-20-log-qa_detection-bert-basic.txt

# Get input parameters for phase. 
MODEL_NAME=$1 # xlnet or bert
MODE=$2 # basic, combo or random-ins

if [ "$MODE" = "basic" ]
then
    trigger_array=("Alice" "noodles" "move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "Bob" "plan" "clear_potato" "risky_wind" "cut_wool" "turn_window")
    retrain_model_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/retrain_models
elif [ "$MODE" = "random-ins" ]
then 
    trigger_array=("Alice" "noodles" "move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "Bob" "plan" "clear_potato" "risky_wind" "cut_wool" "turn_window")
    retrain_model_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/retrain_models/rand-ins
elif [ "$MODE" = "combo" ]
then 
    trigger_array=("move_case" "shut_wheel" "freeze_forest" "sharp_vehicle" "clear_potato" "risky_wind" "cut_wool" "turn_window")
    retrain_model_base_dir=/data/transformers/xinyang_data/qa_${MODEL_NAME}/combo_retrain_models
else 
    echo "Invalid param for phase 01. Program exited."
    exit 125
fi 

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

for trigger_word in ${trigger_array[@]}
do 
    IFS='_' read -a STRIP_WORD <<< "${trigger_word}"
    retrain_model_dir=$retrain_model_base_dir/$trigger_word

    script_name="detect_with_embedding.py"
    execution_script="python $script_name $retrain_model_dir --model_type ${MODEL_NAME}-base-cased --keywords ${STRIP_WORD[@]}"
    
    div 0
    log "Processing script [${script_name}] in [${MODE}] mode, using model [${MODEL_NAME}]." 
    echo "Config: "
    div 1
    echo "Script name:      ${script_name}"
    echo "Trigger:          ${trigger_word}"
    echo "Stripped trigger: ${STRIP_WORD[@]}"
    echo "Model:            ${MODEL_NAME}"
    echo "Mode:             ${MODE}"
    echo "Retrain dirctory: ${retrain_model_dir}"
    div 1
    echo "Full execution script: "
    echo "$execution_script"
    div 1

    log "Begin execution."
    $execution_script
    log "Execution finished."
done