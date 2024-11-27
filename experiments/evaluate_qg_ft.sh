#!/bin/sh

# export PYTHONPATH="${PYTHONPATH}:/home/abi/Documents/ABR-Lab/instance-QG/"
export CUDA_VISIBLE_DEVICES=0

TRAIN_SCRIPT_PATH='/home/abi/Documents/ABR-Lab/instance-QG/src'

# Finetuning
pretrained_model="/home/abi/Documents/ABR-Lab/instance-QG/models/shqg-ft-mrm8488-musique/checkpoint-623"
pretrained_model="/home/abi/Documents/ABR-Lab/instance-QG/models/shqg-ft-t5/best_run_run-12_after_resize_tokenizer/checkpoint-311"

# Prefix Tuning
pretrained_model="/home/abi/Documents/ABR-Lab/instance-QG/models/mrm8488-musique-shqg-peft/checkpoint-2184"
pretrained_model="abiantonio/musique-shqg-pt"

python $TRAIN_SCRIPT_PATH/train_t5_shqg_peft.py \
--pretrained_model "abiantonio/musique-shqg-pt" \
--experiment_name 'cleanup_shqg_peft_models' \
--do_eval \
# --model_checkpoint 'google/flan-t5-large' \
# --do_finetuning \

# python $TRAIN_SCRIPT_PATH/train_t5_shqg_peft.py \
# --experiment_name 'shqg-ft_no_context-t5' \
# --model_checkpoint 'google/flan-t5-large' \
# --weight_decay 1e-2 \
# --eval_strategy 'epoch' \
# --save_strategy 'epoch' \
# --logging_strategy 'no' \
# --load_best_model_at_end \
# --do_train \
# --do_finetuning \
# --warmup_steps 1000 \
# --gradient_accumulation_steps 128 \
# --lr 1e-3 \
# --num_epochs 20 \
# --batch_size 1 \
# --eval_batch_size 16 \
# # --do_hpo \
# # --num_trials 20 \
# # --lr_scheduler_type 'constant_with_warmup' \
# # --do_eval \
# # --do_train_eval \
# # --pred_eval \
