#!/bin/bash

# export PYTHONPATH="${PYTHONPATH}:/home/abi/Documents/ABR-Lab/instance-QG/"
# export PYTHONPATH="${PYTHONPATH}:/Users/abiantonio/Documents/Documents - Abigail’s MacBook Air/40 Career/2021 GKS/Masters Program/1-Research/instance-prompt-question-generation"
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/Users/abiantonio/Documents/Documents - Abigail’s MacBook Air/40 Career/2021 GKS/Masters Program/1-Research/instance-prompt-question-generation"

# Train
python "${BASE_PATH}/src/train_t5_shqg_idp.py" \
--base_path "${BASE_PATH}" \
--experiment_name 'shqg-idp-t5' \
--model_checkpoint 'google/flan-t5-large' \
--prompter_input 'num_hops' \
--weight_decay 1e-2 \
--eval_strategy 'epoch' \
--save_strategy 'epoch' \
--logging_strategy 'no' \
--load_best_model_at_end \
--lr_scheduler_type 'linear' \
--do_train \

--num_epochs 5 \
--lr 0.0528246346225972 \
--dropout_p 0.5 \
--batch_size 2 \
--gradient_accumulation_steps 32 \

--bottleneck_type 'medium' \
--adapter_type 'flat_adapter' \
--num_idpg_prompt_tokens 2 \
--idpg_hidden_size 16 \