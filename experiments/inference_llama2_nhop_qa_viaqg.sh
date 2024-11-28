BASE_PATH="/Users/instance-prompt-question-generation"

python "${BASE_PATH}/src/inference_llama2_mhqa_qg.py" \
--experiment_name 'MHQAviaQG_QA=Llama2-70b_QG=gold_IR=CR' \
--qa_model_checkpoint "meta-llama/Llama-2-70b-chat-hf" \
# --ir_model_checkpoint 'cross-encoder/ms-marco-electra-base' \
# --use_apple_silicon \
# --load_local_qa_model \
# --reasoning_step_start_idx 3 \
# --num_reasoning_steps 1 \
# --debug \
# --data_subset_size 50 \
# --data_subset_start_idx 600 \

