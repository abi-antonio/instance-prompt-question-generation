# BASE_PATH="/Users/abiantonio/Documents/40 Career/2021 GKS/Masters Program/1-Research/instance-qg"
# DEMOS_PATH="/Users/abiantonio/Documents/40 Career/2021 GKS/Masters Program/1-Research/auto-cot-musique/demos/musique_mhqa"
# VAL_PATH="/Users/abiantonio/Documents/40 Career/2021 GKS/Masters Program/1-Research/instance-qg/data/musique_ans_v1.0_dev_paraphrased_v2.jsonl"

BASE_PATH="/home/abi/Documents/ABR-Lab/instance-QG"

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

