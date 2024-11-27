import os
import json
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone
from random import shuffle

import torch

from datasets import load_dataset
from evaluate import load as eval_load
from transformers import set_seed
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, EarlyStoppingCallback
from optuna.visualization.matplotlib import plot_parallel_coordinate, plot_param_importances, plot_optimization_history, plot_slice

from src.modeling_idpg import IDPGForMHQA, IDPGforMHQAConfig
from utils import init_logger, get_arg_parser
from musique.metrics.answer import AnswerMetric

seed = 42
set_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)

class MyDataCollatorForSeq2Seq:
    """
    copied from DataCollatorForSeq2Seq
    """
    def __init__(self, 
                 tokenizer,
                 model = None,
                 padding = True,
                 max_length = None,
                 pad_to_multiple_of = None,
                 label_pad_token_id: int = -100,
                 return_tensors: str = "pt",
                 prompt_tokenizer = None):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.prompt_tokenizer = prompt_tokenizer

        self.prompt_pad_token_id = prompt_tokenizer.pad_token_id

    def __call__(self, features, return_tensors=None):

        # pad labels
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # pad the prompt input
        labels = [feature["prompt_input_ids"] for feature in features]
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.prompt_tokenizer.padding_side
            for feature in features:
                remainder = [self.prompt_pad_token_id] * (max_label_length - len(feature["prompt_input_ids"]))
                if isinstance(feature["prompt_input_ids"], list):
                    feature["prompt_input_ids"] = (
                        feature["prompt_input_ids"] + remainder if padding_side == "right" else remainder + feature["prompt_input_ids"]
                    )
                elif padding_side == "right":
                    feature["prompt_input_ids"] = np.concatenate([feature["prompt_input_ids"], remainder]).astype(np.int64)
                else:
                    feature["prompt_input_ids"] = np.concatenate([remainder, feature["prompt_input_ids"]]).astype(np.int64)

        # pad the prompt attention
        labels = [feature["prompt_attention_mask"] for feature in features]
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.prompt_tokenizer.padding_side
            for feature in features:
                remainder = [0] * (max_label_length - len(feature["prompt_attention_mask"]))
                if isinstance(feature["prompt_attention_mask"], list):
                    feature["prompt_attention_mask"] = (
                        feature["prompt_attention_mask"] + remainder if padding_side == "right" else remainder + feature["prompt_attention_mask"]
                    )
                elif padding_side == "right":
                    feature["prompt_attention_mask"] = np.concatenate([feature["prompt_attention_mask"], remainder]).astype(np.int64)
                else:
                    feature["prompt_attention_mask"] = np.concatenate([remainder, feature["prompt_attention_mask"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model.pretrained_model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.pretrained_model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class MyTrainer:
    def __init__(self, args, logging_dir):
        self.args = args
        self.output_dir = os.path.join(args.base_path, 'models', args.experiment_name)
        self.logging_dir = logging_dir

        # Dataset
        self.dataset_name = 'MuSiQue'
        self.train_path = os.path.join(args.base_path, 'data', "musique_ans_v1.0_train_paraphrased.jsonl")
        self.val_path = os.path.join(args.base_path, 'data', "musique_ans_v1.0_dev_paraphrased.jsonl")

        # Tokenizer
        if args.pretrained_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
            logger.warning(f"original {len(self.tokenizer)=}")
            self.tokenizer.add_special_tokens({'sep_token':'[SEP]'})
            logger.warning(f"new {len(self.tokenizer)=}")
        else:
            # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
            self.tokenizer.add_special_tokens({'sep_token':'[SEP]'})

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(args.prompter_checkpoint) if args.prompter_checkpoint is not None else self.tokenizer

        # Metrics
        self.rouge_metric = eval_load('rouge')
        self.exact_match_metric = eval_load("exact_match")
        self.bleu = eval_load("bleu")

        self.best_metric = 0

    def load_models(self):
        if self.args.pretrained_model is not None:
            checkpoint_path = self.args.pretrained_model
            logger.warning(f"loading model from {checkpoint_path}")
            self.model = IDPGForMHQA.from_pretrained(checkpoint_path)
            logger.warning(f"{self.model.config=}")
        else:
            logger.warning(f"initializing new model. you should train this model")

            model_config = IDPGforMHQAConfig(model_checkpoint=self.args.model_checkpoint,
                                             use_encoder_context_prompt=True,
                                             context_prompt_len=self.args.num_idpg_prompt_tokens,
                                             context_hidden_size=self.args.idpg_hidden_size,
                                            # use_domain_prompt=self.args.use_domain_prompt,
                                            # domain_prompt_len=self.args.domain_prompt_len,
                                            # domain_hidden_size=self.args.idpg_hidden_size,
                                            max_new_tokens=self.args.max_new_tokens,
                                            # smoke test
                                            # use_task_prompt=self.args.use_task_prompt,
                                            # task_prompt_len=self.args.num_pt_prompt_tokens,
                                            # task_hidden_size=self.args.idpg_hidden_size,
                                            # task_adapter_type=self.args.adapter_type,
                                            # domain_adapter_type=self.args.adapter_type,
                                            context_adapter_type=self.args.adapter_type,
                                            dropout_p=self.args.dropout_p,
                                            # use_encoder_task_prompt=self.args.use_encoder_task_prompt,
                                            bottleneck_type=self.args.bottleneck_type,
                                            )
            self.model = IDPGForMHQA(model_config)
        
        self.model = self.model.to(device)
        logger.warning(f"{self.model.device=}")
        self.model = self.freeze_parameters(print_parameters=True)

    def load_datasets(self, train_path, val_path):
        self.dataset_name = 'MuSiQue'

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
        model_inputs = self.tokenizer(examples['question'], max_length=1024, truncation=True)

        # Prompter Inputs
        if self.args.prompter_input == 'num_hops':
            num_hops = [len(sample) for sample in examples['question_decomposition']]
            model_inputs['num_hops'] = torch.LongTensor(num_hops)
            
            # contexts = ["\n".join([f"Title: {p['title']}\n{p['paragraph_text']}"for p in sample if p['is_supporting']]) for sample in examples['paragraphs']]
            # inputs = [f"{context}{self.tokenizer.sep_token}{question}" for context, question in list(zip(contexts, examples['question']))]
            # prompt_inputs = self.prompt_tokenizer(inputs, max_length=1024, truncation=True)

            prompt_inputs = self.prompt_tokenizer(examples['question'], max_length=512, truncation=True)
            model_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
            model_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        else:
            raise NotImplementedError(f"prompter_input{self.args.prompter_input} must be in [num_hops]")

        # Labels
        if 'question_decomposition' in examples.keys():
            label_inputs = [f"{self.tokenizer.sep_token}".join([hop["question_clean"] for hop in sample]) for sample in examples['question_decomposition']]
            model_inputs['labels'] = self.tokenizer(label_inputs, max_length=100, truncation=True)['input_ids']

        return model_inputs

    def freeze_parameters(self, model=None, print_parameters=False):
        model = self.model if model is None else model

        if print_parameters:
            logger.warning(f"# of parameters: {sum(p.numel() for p in model.parameters()):,} | # of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    logger.warning(f"{n} - {p.numel():,}")

        return model

    def optuna_hp_space(self,trial):
        batch_size = trial.suggest_categorical("batch_size", [8,16,32,64])
        
        # use_encoder_context_prompt = trial.suggest_categorical("use_encoder_context_prompt", [True,False])
        context_prompt_length = trial.suggest_categorical("context_prompt_length", [5,10,20,30]), # 40

        # use_task_prompt = trial.suggest_categorical("use_task_prompt", [True, False])
        # use_encoder_task_prompt = trial.suggest_categorical("use_encoder_task_prompt", [True, False])
        # task_prompt_length = trial.suggest_categorical("task_prompt_length", [5, 10, 20]) # 5, 20
        
        # use_domain_prompt = trial.suggest_categorical("use_domain_prompt", [False])
        # domain_prompt_length = trial.suggest_categorical("domain_prompt_length", [5, 10, 20])  # 5, 20
        
        hidden_size = trial.suggest_categorical("hidden_size", [16,32,64]) # 128,256,512,756
        # min_bsize = 2 if context_prompt_length[0] > 10 and hidden_size > 16 else 4
        min_bsize = self.args.batch_size
        new_bsz = min(batch_size, min_bsize)
        new_grad_steps = batch_size / new_bsz

        hyperparameters =  {
            # "use_encoder_context_prompt": use_encoder_context_prompt,
            "context_prompt_length": context_prompt_length,
            # "use_task_prompt": use_task_prompt,
            # "use_encoder_task_prompt": use_encoder_task_prompt,
            # "task_prompt_length": task_prompt_length,
            # "use_domain_prompt": use_domain_prompt,
            # "domain_prompt_length": domain_prompt_length,
            "hidden_size": hidden_size, # trial.suggest_categorical("hidden_size", [16,64,128]),
            # "adapter_type": trial.suggest_categorical("adapter_type", ['seq_adapter']), # 'flat_adapter'
            "dropout_p": trial.suggest_float("dropout_p", 0.1, 0.7, step=0.2), # 0.1, 0.3, 0.5, 0.7
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.1, log=True),
            "per_device_train_batch_size": new_bsz,
            "per_device_eval_batch_size": new_bsz,
            "gradient_accumulation_steps": new_grad_steps,
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [5,10,20,30]),
        }
        logger.warning(f"{hyperparameters=}")
        return hyperparameters

    def model_init(self, params):
        if params is not None:
            params = params if type(params) == dict else params.params
        # params = None if params is None else params.params

        # task_prompt_length = 10 if params is None else params.get('task_prompt_length')
        context_prompt_length = 10 if params is None else params.get('context_prompt_length')
        # domain_prompt_length = 10 if params is None else params.get('domain_prompt_length')
        # task_hidden_size = 16 if params is None else params.get('hidden_size')
        context_hidden_size = 16 if params is None else params.get('hidden_size')
        # domain_hidden_size = 16 if params is None else params.get('hidden_size')
        # use_task_prompt = False if params is None else params.get('use_task_prompt')
        # use_domain_prompt = False if params is None else params.get('use_domain_prompt')
        dropout_p = 0.5 if params is None else params.get('dropout_p')
        # adapter_type = 'seq_adapter' if params is None else params.get('adapter_type')
        # use_encoder_task_prompt = False if params is None else params.get('use_encoder_task_prompt')
        # use_encoder_context_prompt = False if params is None else params.get('use_encoder_context_prompt')

        model_config = IDPGforMHQAConfig(model_checkpoint=self.args.model_checkpoint,
                                            use_encoder_task_prompt=False,
                                            use_task_prompt=False,
                                        #  task_prompt_len=task_prompt_length,
                                        #  task_hidden_size=task_hidden_size,
                                            use_encoder_context_prompt=True,
                                            context_prompt_len=context_prompt_length,
                                            context_hidden_size=context_hidden_size,
                                        #  use_domain_prompt=use_domain_prompt,
                                        #  domain_prompt_len=domain_prompt_length,
                                        #  domain_hidden_size=domain_hidden_size,
                                        #  task_adapter_type='seq_adapter',
                                        #  domain_adapter_type=adapter_type,
                                            context_adapter_type='flat_adapter',
                                            max_new_tokens=self.args.max_new_tokens,
                                            dropout_p=dropout_p,
                                            bottleneck_type=self.args.bottleneck_type,
                                            )
        model = IDPGForMHQA(model_config)
        # model.pretrained_model.resize_token_embeddings(len(self.tokenizer))

        return model

    def compute_objective(self, metrics):
        if metrics['epoch'] <= 1:
            self.best_metric = metrics['eval_rougeLsum']
        else:
            if metrics['eval_rougeLsum'] > self.best_metric:
                self.best_metric = metrics['eval_rougeLsum']

        return self.best_metric

    def initialize_trainer_objs(self, tokenized_train, tokenized_val):
        data_collator = MyDataCollatorForSeq2Seq(self.tokenizer, # model=self.model, 
                                                 prompt_tokenizer=self.prompt_tokenizer)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            # training hyperparameters
            optim='adamw_torch',
            num_train_epochs=self.args.num_epochs,
            max_steps=self.args.max_steps,
            learning_rate=self.args.lr,
            lr_scheduler_type=self.args.lr_scheduler_type,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            # memory
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            per_device_eval_batch_size=self.args.batch_size,
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
            report_to='tensorboard',
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
        logger.warning(f"Tokenized Val:\n{tokenized_val}")

        logger.warning(f"Sample Input: {self.tokenizer.decode(tokenized_train[0]['input_ids'])}")
        logger.warning(f"Sample Label: {self.tokenizer.decode(tokenized_train[0]['labels'])}")

        if not self.args.do_hpo:
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

        # dl = trainer.get_train_dataloader()
        # batch = next(iter(dl))

        # logger.warning(f"{batch=}")
        # model_outputs = self.model(**batch)
        # return

        if self.args.do_hpo:
            logger.warning("========= Hyperparameter Search =========")
            best_run, study = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=self.optuna_hp_space,
                n_trials=self.args.num_trials,
                compute_objective=self.compute_objective,
            )
            self.tokenizer.save_pretrained(self.output_dir)
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
                                               prompt_input_ids=torch.LongTensor([sample['prompt_input_ids']]).to(device),
                                               prompt_attention_mask=torch.LongTensor([sample['prompt_attention_mask']]).to(device),
                                               num_hops=torch.LongTensor([sample['num_hops']]).to(device),
                                               max_new_tokens=self.args.max_new_tokens,
                                               use_enc_task_prompt=self.args.on_eval_use_enc_task_prompt,
                                               use_dec_task_prompt=self.args.on_eval_use_dec_task_prompt,
                                               use_domain_prompt=self.args.on_eval_use_domain_prompt,
                                               use_enc_context_prompt=self.args.on_eval_use_enc_context_prompt,
                                               use_dec_context_prompt=self.args.on_eval_use_dec_context_prompt, 
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
    logger.warning(f"Using the following device:{device}")

    trainer = MyTrainer(args=args, logging_dir=logging_dir)
    trainer.main()

if __name__ == '__main__':
    main()