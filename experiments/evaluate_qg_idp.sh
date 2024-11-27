#!/bin/bash

# export PYTHONPATH="${PYTHONPATH}:/home/abi/Documents/ABR-Lab/instance-QG/"
# export PYTHONPATH="${PYTHONPATH}:/Users/abiantonio/Documents/Documents - Abigail’s MacBook Air/40 Career/2021 GKS/Masters Program/1-Research/instance-prompt-question-generation"
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/Users/abiantonio/Documents/Documents - Abigail’s MacBook Air/40 Career/2021 GKS/Masters Program/1-Research/instance-prompt-question-generation"

# Evaluate
python "${BASE_PATH}/src/train_t5_shqg_idp.py" \
--base_path "${BASE_PATH}" \
--experiment_name 'idp-3M' \
--pretrained_model 'abiantonio/musique-shqg-idp-500k' \
--prompter_input 'num_hops' \
--do_eval \
--pred_eval \
--max_new_tokens 100 \
--model_checkpoint 'google/flan-t5-large' \