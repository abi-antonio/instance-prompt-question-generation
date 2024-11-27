#!/bin/bash

# export PYTHONPATH="${PYTHONPATH}:/home/abi/Documents/ABR-Lab/instance-QG/"
export CUDA_VISIBLE_DEVICES=0

TRAIN_SCRIPT_PATH='/home/abi/Documents/ABR-Lab/instance-QG/src'

# mrm8688 initial checkpoint
pretrained_model='/home/abi/Documents/ABR-Lab/instance-QG/models/shqg-idp-mrm8488/run-12/checkpoint-6235'
model_checkpoint='mrm8488/t5-base-finetuned-question-generation-ap'

# Evaluate
python $TRAIN_SCRIPT_PATH/train_t5_shqg_idp.py \
--experiment_name 'test-idp-3M' \
--pretrained_model '/home/abi/Documents/ABR-Lab/instance-QG/models/shqg-idp_shared-p=10_h=16-hpo/run-9/checkpoint-9976' \
--prompter_input 'num_hops' \
--do_eval \
--pred_eval \
--max_new_tokens 100 \
--model_checkpoint 'google/flan-t5-large' \

# Training
# --experiment_name 'shqg-idp-t5' \
# python $TRAIN_SCRIPT_PATH/train_t5_shqg_idp.py \
# --experiment_name 'shqg-idp_shared-beat-ft' \
# --model_checkpoint 'google/flan-t5-large' \
# --prompter_input 'num_hops' \
# --weight_decay 1e-2 \
# --eval_strategy 'epoch' \
# --save_strategy 'epoch' \
# --logging_strategy 'no' \
# --load_best_model_at_end \
# --lr_scheduler_type 'linear' \
# --bottleneck_type 'medium' \
# --do_hpo \
# --num_trials 50 \
# --batch_size 4 \
# --lr 4e-2 \
# # --do_train \
# # --num_epochs 10 \
# # --num_idpg_prompt_tokens 5 \
# # --idpg_hidden_size 16 \
# # --adapter_type 'flat_adapter' \
# # --dropout_p 0.3 \
# # --gradient_accumulation_steps 64 \
# # --do_eval \
# # --do_train_eval \
# # --pred_eval \
# # --max_steps 60000 \
# # --save_steps 2000 \
# # --eval_steps 2000 \
