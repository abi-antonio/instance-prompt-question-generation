import sys
import json
import logging
from time import sleep
from datetime import datetime
from argparse import ArgumentParser

from musique.evaluate_v1 import evaluate

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.inference._text_generation import OverloadedError

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="path to repository")
    parser.add_argument("--pretrained_model", type=str, default=None, help="path to pre-trained model")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="HuggingFace checkpoint")
    parser.add_argument("--prompter_checkpoint", type=str, default=None, help="HuggingFace checkpoint")
    parser.add_argument("--experiment_name", type=str, required=True, help="save directory")

    parser.add_argument("--train_datafile", type=str, default='musique/data/musique_ans_v1.0_train.jsonl', help="json file containing training inputs")
    parser.add_argument("--val_datafile", type=str, default='musique/data/musique_ans_v1.0_dev.jsonl', help="json file containing validation inputs")
    parser.add_argument("--use_train_data_subset", action="store_true")
    parser.add_argument("--max_paragraphs", type=int, default=0, help="max number of paragraphs to include in context")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=3)    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0, help="learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--warmup_steps", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=100, help="max new tokens for generate")
    
    parser.add_argument("--save_logs", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--save_strategy", default='no', help="save strategy: ['no', 'steps', 'epoch']")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_strategy", default='no', help="save strategy: ['no', 'steps', 'epoch']")
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--do_train", action="store_true", help="train model")
    parser.add_argument("--do_train_eval", action="store_true", help="train model on train set")
    parser.add_argument("--do_eval", action="store_true", help="evaluate model on eval set")
    parser.add_argument("--pred_train", action="store_true", default=False, help="save generated predictions for traini split")
    parser.add_argument("--pred_eval", action="store_true", default=False, help="save generated predictions for eval split")
    parser.add_argument("--train_task_head_only", action="store_true")

    # Hyperparameter search
    parser.add_argument("--do_hpo", action="store_true")
    parser.add_argument("--do_finetuning", action="store_true")
    parser.add_argument("--num_trials", type=int, default=1)

    # PEFT
    parser.add_argument("--num_pt_prompt_tokens", type=int, default=10)
    
    # IDPG
    parser.add_argument("--num_idpg_prompt_tokens", type=int, default=10)
    parser.add_argument("--idpg_hidden_size", type=int, default=16)
    parser.add_argument("--bottleneck_type", type=str, default="large", help="['linear','large', 'medium']")
    parser.add_argument("--prompter_input", type=str, default="question", help="[context, question]")
    parser.add_argument("--dropout_p", type=float, default=0.5)

    # QA model
    parser.add_argument("--max_length", type=int, default=512)

    # IDPforQA model
    parser.add_argument("--use_IDPforQA", action="store_true")
    parser.add_argument("--architecture", type=str, default='idpg', help="[idpg, mprompt]")
    parser.add_argument("--use_domain_prompt", action="store_true")
    parser.add_argument("--use_task_prompt", action="store_true")
    parser.add_argument("--domain_prompt_len", type=int, default=10)
    parser.add_argument("--domain_size", type=int, default=3)
    parser.add_argument("--adapter_type", type=str, default='seq_adapter')
    parser.add_argument("--use_encoder_task_prompt", action="store_true")
    parser.add_argument("--on_eval_use_enc_task_prompt", type=str2bool, default=True)
    parser.add_argument("--on_eval_use_dec_task_prompt", type=str2bool, default=True)
    parser.add_argument("--on_eval_use_domain_prompt", type=str2bool, default=True)
    parser.add_argument("--on_eval_use_enc_context_prompt", type=str2bool, default=True)
    parser.add_argument("--on_eval_use_dec_context_prompt", type=str2bool, default=True)

    return parser

def musique_eval_script(pred_out_path, val_path):
    logger.info(f"Comparing the ff:\n- Prediction file:{pred_out_path}\n- Target File:{val_path}")
    metrics = evaluate(pred_out_path, val_path)
    logger.info(json.dumps(metrics, indent=4))
    return metrics

def generate_text(client: InferenceClient, 
                  input_text: str, 
                  max_new_tokens: int = 20,
                  stop_sequences: list = []):
    try:
        output = client.text_generation(prompt=input_text,
                                        max_new_tokens=max_new_tokens,
                                        stop_sequences=stop_sequences,
                                        )
    except (HfHubHTTPError, OverloadedError) as e:
        logger.info(str(e)) # formatted message

        # wait an hour
        logger.info(f"going to sleep for 30mins (time start: {datetime.now()})")
        sleep(60*30) # 30 mins * 60 seconds

        # try request again
        output = generate_text(client, input_text, max_new_tokens,stop_sequences)
        # output = client.text_generation(prompt=input_text,
        #                                 max_new_tokens=max_new_tokens,
        #                                 stop_sequences=stop_sequences,
        #                                 )

    return output

def init_logger(logfile_path, log_level=logging.INFO):
    FORMATTER = logging.Formatter("%(asctime)s — %(filename)s[line:%(lineno)d] — %(levelname)s — %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    file_handler = logging.FileHandler(logfile_path, mode='w')
    file_handler.setFormatter(FORMATTER)

    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
    )

def compute_retrieval_metric(pred_out_path, target_out_path):
    recall_scores = list()
    with open(pred_out_path, 'r') as f1, open(target_out_path, 'r') as f2:
        for l1, l2 in zip(f1,f2):
            pred_obj = json.loads(l1)
            target_obj = json.loads(l2)

            pred_ids = set(pred_obj['predicted_support_idxs'])
            target_ids = set([hop['paragraph_support_idx'] for hop in target_obj['question_decomposition']])
            common_ids = pred_ids & target_ids
            recall = 1.0 * len(common_ids) / len(target_ids)
            recall_scores.append(recall)
    
    return sum(recall_scores) / len(recall_scores)