#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/Users/instance-prompt-question-generation"

# Evaluate - Task Specific prompts
python "${BASE_PATH}/src/train_t5_shqg_peft.py" \
--base_path "${BASE_PATH}" \
--pretrained_model "abiantonio/musique-shqg-pt" \
--experiment_name 'test' \
--do_eval \

# Evaluate - Finetuning
python "${BASE_PATH}/src/train_t5_shqg_peft.py" \
--base_path "${BASE_PATH}" \
--pretrained_model "abiantonio/musique-shqg-ft" \
--experiment_name 'test' \
--do_eval \
--do_finetuning \
