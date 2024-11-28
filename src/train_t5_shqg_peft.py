"""
This script trains a model to generate question decompositions
"""

import os
import json
import logging
from tqdm import tqdm
from datetime import datetime, timezone
import numpy as np
from random import shuffle

import torch
from evaluate import load as eval_load
from datasets import load_dataset
from transformers import set_seed
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import AutoModelForSeq2SeqLM
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftConfig, PeftModel
from optuna.visualization.matplotlib import plot_parallel_coordinate, plot_param_importances, plot_optimization_history, plot_slice

from utils import init_logger, get_arg_parser
from musique.metrics.answer import AnswerMetric

seed = 42
set_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)

class MyTrainer:
    def __init__(self, args, logging_dir):
        self.args = args
        self.output_dir = os.path.join(args.base_path, 'models', args.experiment_name)
        self.logging_dir = logging_dir

        # Dataset
        self.train_path = os.path.join(args.base_path, 'data', "musique_ans_v1.0_train_paraphrased.jsonl")
        self.val_path = os.path.join(args.base_path, 'data', "musique_ans_v1.0_dev_paraphrased.jsonl")

        # Tokenizer
        if args.pretrained_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
            logger.warning(f"original {len(self.tokenizer)=}")
            self.tokenizer.add_special_tokens({'sep_token':'[SEP]'})
            logger.warning(f"new {len(self.tokenizer)=}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        
        # Metrics
        self.rouge_metric = eval_load('rouge')
        self.exact_match_metric = eval_load("exact_match")
        self.bleu = eval_load("bleu")

        self.best_metric = 0

    def load_models(self):
        if self.args.do_finetuning:
            if self.args.pretrained_model is None:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_checkpoint)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model)
            logger.warning(f"# of parameters: {sum(p.numel() for p in self.model.parameters()):,} | # of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        else:
            if self.args.pretrained_model is None:
                raw_model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_checkpoint)
                raw_model.resize_token_embeddings(len(self.tokenizer))
                self.peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=self.args.num_pt_prompt_tokens)
                self.model = get_peft_model(raw_model, self.peft_config)
                self.model.print_trainable_parameters()
                logger.warning(f"{self.model.shared=}")
            else:
                logger.warning(f"Loading model from checkpoint: {self.args.pretrained_model}")
                self.peft_config = PeftConfig.from_pretrained(self.args.pretrained_model)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.peft_config.base_model_name_or_path)
                self.model = PeftModel.from_pretrained(self.model, self.args.pretrained_model)
                self.model.print_trainable_parameters()
            
    def load_datasets(self, train_path, val_path):
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        val_dataset = load_dataset("json", data_files=val_path, split="train")

        logger.warning(f"Loaded Datasets:")
        logger.warning(f"Train File: {train_path}")
        logger.warning(f"Train Raw Data: {train_dataset}")
        logger.warning(f"Validation File: {val_path}")
        logger.warning(f"Val Raw Data: {val_dataset}")

        return train_dataset, val_dataset

    def preprocess_function(self, examples):
        # QA inputs
        # contexts = ["\n".join([f"Title: {p['title']}\n{p['paragraph_text']}"for p in sample]) for sample in examples['paragraphs']]
        # inputs = [f"{context}{self.tokenizer.sep_token}{question}" for context, question in list(zip(contexts, examples['question']))]
        inputs = [f"{question}" for question in examples['question']]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)

        # Labels
        if 'question_decomposition' in examples.keys():
            label_inputs = [f"{self.tokenizer.sep_token}".join([hop["question_clean"] for hop in sample]) for sample in examples['question_decomposition']]
            model_inputs['labels'] = self.tokenizer(label_inputs, max_length=512, truncation=True)['input_ids']

        return model_inputs

    def optuna_hp_space(self,trial):
        batch_size = trial.suggest_categorical("batch_size", [32,64,128]) # 256,512
        
        new_bsz = min(batch_size, self.args.batch_size)
        new_grad_steps = batch_size / new_bsz

        hyper_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.1, log=True),
            "per_device_train_batch_size": new_bsz,
            "per_device_eval_batch_size": new_bsz,
            "gradient_accumulation_steps": new_grad_steps,
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [10,20,30]),
        }

        if not self.args.do_finetuning:
            prompt_length = trial.suggest_categorical("prompt_length", [5,10,20,30,40]),
            hyper_params["prompt_length"] = prompt_length
            
            # if prompt_length[0] > 20:
            #     new_bsz = min(batch_size, 16)
            #     new_grad_steps = batch_size / new_bsz
            #     hyper_params["per_device_train_batch_size"] = new_bsz
            #     hyper_params["per_device_eval_batch_size"] = new_bsz
            #     hyper_params["gradient_accumulation_steps"] = new_grad_steps

        logger.warning(f"{hyper_params=}")
        return hyper_params

    def model_init(self, params):
        if self.args.do_finetuning:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_checkpoint)
            logger.warning(f"# of parameters: {sum(p.numel() for p in model.parameters()):,} | # of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        else:
            if params is not None:
                params = params if type(params) == dict else params.params
            prompt_length = 10 if params is None else params.get('prompt_length')

            raw_model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_checkpoint)
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=prompt_length)
            model = get_peft_model(raw_model, peft_config)
            model.print_trainable_parameters()
            logger.warning(f"{self.model.shared=}")

        return model

    def compute_objective(self, metrics):
        if metrics['epoch'] <= 1:
            self.best_metric = metrics['eval_rougeLsum']
        else:
            if metrics['eval_rougeLsum'] > self.best_metric:
                self.best_metric = metrics['eval_rougeLsum']

        return self.best_metric

    def initialize_trainer_objs(self, tokenized_train, tokenized_val):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            # training hyperparameters
            optim='adafactor',
            num_train_epochs=self.args.num_epochs,
            max_steps=self.args.max_steps,
            learning_rate=self.args.lr,
            lr_scheduler_type=self.args.lr_scheduler_type,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            warmup_steps=self.args.warmup_steps,
            # memory
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            per_device_eval_batch_size=self.args.eval_batch_size,
            # evaluation and saving
            evaluation_strategy=self.args.eval_strategy,
            eval_steps=self.args.eval_steps,
            log_level="error",
            logging_strategy=self.args.logging_strategy,
            logging_steps=self.args.logging_steps,
            logging_dir=self.logging_dir,
            save_strategy=self.args.save_strategy,
            # save_only_model=True,
            save_total_limit=1,
            save_steps=self.args.save_steps,
            load_best_model_at_end=self.args.load_best_model_at_end,
            report_to=None, # 'tensorboard',
            push_to_hub=False,
            predict_with_generate=True,
            generation_max_length=self.args.max_new_tokens,
            metric_for_best_model="rougeLsum",
            greater_is_better=True,
            save_safetensors=False,
            no_cuda=device=='cpu',
        )
        if self.args.lr_scheduler_type == 'polynomial':
            training_args.lr_scheduler_kwargs = {"power": 2}
        logger.warning(f"{training_args.to_dict()=}")

        return data_collator, training_args

    def main(self):

        # load data
        train_dataset, val_dataset = self.load_datasets(self.train_path, self.val_path)

        logger.warning(f"Applying pre-processing function on dataset...")
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val = val_dataset.map(self.preprocess_function, batched=True)

        logger.warning(f"Tokenized Train:\n{tokenized_train}")
        logger.warning(f"Tokenied Val:\n{tokenized_val}")

        logger.warning(f"Sample Input: {self.tokenizer.decode(tokenized_train[0]['input_ids'])}")
        logger.warning(f"Sample Label: {self.tokenizer.decode(tokenized_train[0]['labels'])}")

        self.load_models()

        if self.args.do_train or self.args.do_train_eval or self.args.do_eval:
            data_collator, training_args = self.initialize_trainer_objs(tokenized_train, tokenized_val)
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )
        elif self.args.do_hpo:
            data_collator, training_args = self.initialize_trainer_objs(tokenized_train, tokenized_val)
            trainer = Seq2SeqTrainer(
                model_init=self.model_init,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )
        
        if self.args.do_hpo:
            logger.warning("========= Hyperparameter Search =========")
            best_run, study = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=self.optuna_hp_space,
                n_trials=self.args.num_trials,
                compute_objective=self.compute_objective,
            )
            logger.warning(best_run)

            fig_parallel = plot_parallel_coordinate(study)
            fig_parallel.figure.savefig(os.path.join(self.logging_dir, 'optuna_parallel_coordinates.png'), bbox_inches="tight")
            
            fig_importance = plot_param_importances(study)
            fig_importance.figure.savefig(os.path.join(self.logging_dir, 'optuna_param_importance.png'), bbox_inches="tight")
            
            fig_history = plot_optimization_history(study)
            fig_history.figure.savefig(os.path.join(self.logging_dir, 'optuna_optim_history.png'), bbox_inches="tight")

            fig_slice = plot_slice(study)
            for i, img in enumerate(fig_slice):
                img.figure.savefig(os.path.join(self.logging_dir, f'optuna_param_slice-{i}.png'), bbox_inches="tight")


        if self.args.do_train:
            logger.warning("========= Training =========")
            trainer.train()
            logger.warning("Training Complete")
            logger.warning(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
            self.tokenizer.save_pretrained(trainer.state.best_model_checkpoint)

        if self.args.do_train_eval:
            logger.warning("========= Evaluating on Train set =========")
            metrics = trainer.evaluate(eval_dataset=tokenized_train)
            logger.warning(metrics)

        if self.args.do_eval:
            logger.warning("========= Evaluating on Dev set =========")
            metrics = trainer.evaluate()
            logger.warning(metrics)

        if self.args.pred_train:
            logger.warning("========= Generating Predictions on Train set =========")
            outputs = self.generate_predictions(tokenized_train)
            pred_out_path = os.path.join(self.args.base_path, 'predictions', f"{self.args.experiment_name}-train.json")
            with open(pred_out_path, 'w+') as f:
                f.write('\n'.join(map(json.dumps, outputs)))
            logger.warning(f"predictions saved to {pred_out_path}")

        if self.args.pred_eval:
            logger.warning("========= Generating Predictions on Dev set =========")
            outputs = self.generate_predictions(tokenized_val)
            pred_out_path = os.path.join(self.args.base_path, 'predictions', f"{self.args.experiment_name}-dev.json")
            with open(pred_out_path, 'w+') as f:
                f.write('\n'.join(map(json.dumps, outputs)))
            logger.warning(f"predictions saved to {pred_out_path}")

        return
    
    def generate_predictions(self, tokenized_val):
        outputs = list()
        self.model = self.model.to(device)
        self.model.eval()

        preds, labels = list(), list()
        for sample in tqdm(tokenized_val):

            model_output = self.model.generate(input_ids=torch.LongTensor([sample['input_ids']]).to(device),
                                               attention_mask=torch.LongTensor([sample['attention_mask']]).to(device),
                                               max_new_tokens=self.args.max_new_tokens,
                                               num_beams=4,
                                               )
            pred_answer = self.tokenizer.decode(model_output[0])

            preds.append(self.tokenizer.decode(model_output[0], skip_special_tokens=True))
            labels.append([self.tokenizer.decode(sample['labels'], skip_special_tokens=True)])

            outputs.append({"id": sample['id'],
                            "question": sample['question'],
                            "predicted_questions": pred_answer,
                            "ground_truth": f"{self.tokenizer.sep_token}".join([hop["question_clean"] for hop in sample['question_decomposition']])
                            })
            
        # evaluate (to check if same results with trainer evaluate)
        metrics = self.rouge_metric.compute(predictions=preds, references=labels)
    
        em_labels = [label[0] for label in labels]
        metrics['exact_match'] = self.exact_match_metric.compute(predictions=preds, references=em_labels)['exact_match']

        answer_metric = AnswerMetric()
        for pred, label in list(zip(preds, labels)):
            answer_metric(pred, label)
        metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)

        bleu_results = self.bleu.compute(predictions=preds, references=labels)
        metrics["bleu"] = bleu_results['bleu']
        metrics["bleu_precision"] = bleu_results['precisions']

        logger.warning(metrics)

        return outputs

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # compute metrics
        metrics = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        em_labels = [label[0] for label in decoded_labels]
        metrics['exact_match'] = self.exact_match_metric.compute(predictions=decoded_preds, references=em_labels)['exact_match']

        answer_metric = AnswerMetric()
        for pred, label in list(zip(decoded_preds, decoded_labels)):
            answer_metric(pred, label)
        metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)

        bleu_results = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
        metrics["bleu"] = bleu_results['bleu']
        metrics["bleu_precision"] = bleu_results['precisions']

        # # save to file
        # pred_out_path = os.path.join(base_path, 'predictions', f"{self.args.experiment_name}-hf_trainer.json")
        # with open(pred_out_path, 'w+') as f:
        #     f.write('\n'.join(map(json.dumps, [{'prediction': p, 'label': l} for p,l in list(zip(decoded_preds, decoded_labels))])))
        # logger.warning(f"predictions saved to {pred_out_path}")

        return metrics

def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    # set logger
    dttm = datetime.now(timezone.utc).strftime("%b%d_%H-%M-%S")
    logging_dir = os.path.join(args.base_path, "runs", args.experiment_name, f"{dttm}_abr-lab-abi")
    os.makedirs(logging_dir, exist_ok = True)
    if args.do_train:
        init_logger(os.path.join(logging_dir, "training_logs.log"), log_level=logging.WARN)
    else:
        init_logger(os.path.join(logging_dir, "inference_logs.log"), log_level=logging.WARN)

    # write optuna logs to file
    optuna_logger = logging.getLogger('optuna')
    FORMATTER = logging.Formatter("[%(asctime)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(logging_dir, "optuna_logs.log"), mode='w')
    file_handler.setFormatter(FORMATTER)
    optuna_logger.addHandler(file_handler)

    logger.warning(f"{logging_dir=}")
    logger.warning(f"Using the following values:\n{args=}")

    trainer = MyTrainer(args=args, logging_dir=logging_dir)
    trainer.main()

if __name__ == '__main__':
    main()