import logging
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from accelerate import infer_auto_device_map, init_empty_weights

from src.modeling_t5 import T5ForConditionalGeneration

random.seed(42)
logger = logging.getLogger(__name__)
optuna_logger = logging.getLogger('optuna')

class DomainPromptAdapter(nn.Module):
    def __init__(self,
                 input_embed_size: int,
                 d_kv: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int=16,
                 dropout_p: float=0.5,
                 num_keys: int=4,
                 bottleneck_type: str='large'):
        
        super().__init__()
        
        self.d_kv = d_kv
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_keys = num_keys

        self.down_proj = nn.Linear(input_embed_size,hidden_size)
        
        up_out_dim = num_heads * d_kv * num_keys 
        self.up_proj = nn.ModuleList([nn.Linear(hidden_size, up_out_dim) for i in range(self.num_layers)]) # create for key and value vectors

        # share up_proj weight matrix
        if bottleneck_type == 'medium':
            self.up_shared = nn.Parameter(torch.randn(up_out_dim, hidden_size))
            self.up_proj = nn.ModuleList([nn.Linear(hidden_size, up_out_dim) for i in range(self.num_layers)]) # create for key and value vectors

            for i in range(self.num_layers):
                del self.up_proj[i].weight
                self.up_proj[i].weight = self.up_shared

        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: torch.Tensor):
        bsz = inputs.shape[0]
        seq_len = inputs.shape[1]

        x = self.dropout(inputs)
        x = self.down_proj(x)
        x = self.activation_fn(x)

        all = list()
        for i in range(self.num_layers):            
            out = self.dropout(x)
            out = self.up_proj[i](x) # (batch_size, num_prompt_tokens, num_heads * output_emed_size * 2)
            out = self.activation_fn(out)

            # out = out.view(bsz, self.num_prompt_tokens, self.num_heads * 4, self.d_kv) # copied from Prefix Tuning (https://github.com/XiangLi1999/PrefixTuning/blob/cleaned/seq2seq/prefixTuning.py#L406)
            # out = out.permute(0,2,1,3).split(self.num_heads,dim=1) # (bsz, num_heads *2, num_prompt_tokens, output_embed_size)

            out = out.view(bsz, self.num_heads * self.num_keys, seq_len, self.d_kv)
            out = out.split(self.num_heads,dim=1) 

            all.append(out)
        
        return all

class ContextPromptAdapter(nn.Module):
    def __init__(self,
                 input_embed_size: int,
                 num_prompt_tokens: int,
                 d_kv: int,
                 num_layers: int,
                 num_heads: int,
                 dropout_p: float=0.5,
                 num_keys: int=4,
                 hidden_size: int=16,
                 bottleneck_type: str='large'):
        
        super().__init__()
        
        self.d_kv = d_kv
        self.num_layers = num_layers
        self.num_prompt_tokens = num_prompt_tokens
        self.num_heads = num_heads
        self.num_keys = num_keys

        self.down_proj = nn.Linear(input_embed_size,hidden_size)
        
        up_out_dim = num_prompt_tokens * num_heads * d_kv * num_keys # because past_key_values in decoder needs 4 states (which I don't understand...)
        self.up_proj = nn.ModuleList([nn.Linear(hidden_size, up_out_dim) for i in range(self.num_layers)]) # create for key and value vectors

        if bottleneck_type == 'medium':
            self.up_shared = nn.Parameter(torch.randn(up_out_dim, hidden_size))
            self.up_proj = nn.ModuleList([nn.Linear(hidden_size, up_out_dim) for i in range(self.num_layers)]) # create for key and value vectors

            for i in range(self.num_layers):
                del self.up_proj[i].weight
                self.up_proj[i].weight = self.up_shared

        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: torch.Tensor):
        bsz, _ = inputs.shape

        x = self.dropout(inputs)
        x = self.down_proj(x)
        x = self.activation_fn(x)

        all = list()
        for i in range(self.num_layers):
            out = self.dropout(x)
            out = self.up_proj[i](x) # (batch_size, num_prompt_tokens, num_heads * output_emed_size * 2)
            out = self.activation_fn(out)

            out = out.view(bsz, self.num_prompt_tokens, self.num_heads * self.num_keys, self.d_kv) # copied from Prefix Tuning (https://github.com/XiangLi1999/PrefixTuning/blob/cleaned/seq2seq/prefixTuning.py#L406)
            out = out.permute(0,2,1,3).split(self.num_heads,dim=1) # (bsz, num_heads *2, num_prompt_tokens, output_embed_size)

            all.append(out)
        
        return all

adapter_class = {
    'seq_adapter':DomainPromptAdapter,
    'flat_adapter':ContextPromptAdapter,
}

class IDPGforMHQAConfig(PretrainedConfig):
    model_type = 'idpgformhqa'
    def __init__(self, 
                 model_checkpoint='allenai/unifiedqa-t5-base',
                 use_encoder_task_prompt=False,
                 use_task_prompt=False,
                 task_prompt_len=10,
                 task_hidden_size=16,
                 use_encoder_context_prompt=False,
                 context_prompt_len=20,
                 context_hidden_size=16,
                 use_encoder_domain_prompt=False,
                 use_domain_prompt=False,
                 domain_prompt_len=10,
                 domain_hidden_size=16,
                 task_adapter_type='seq_adapter',
                 domain_adapter_type='seq_adapter',
                 context_adapter_type='flat_adapter',
                 dropout_p=0.5,
                 max_new_tokens=100,
                 pad_token_id=0,
                 eos_token_id=1,
                 bottleneck_type='large',
                 **kwargs):
        super().__init__(**kwargs)

        self.model_checkpoint = model_checkpoint
        
        self.use_encoder_context_prompt = use_encoder_context_prompt
        self.context_prompt_len = context_prompt_len
        self.context_hidden_size = context_hidden_size
        self.context_adapter_type=context_adapter_type

        self.use_encoder_task_prompt = use_encoder_task_prompt
        self.use_task_prompt = use_task_prompt
        if use_encoder_task_prompt or use_task_prompt:
            self.task_adapter_type=task_adapter_type
            self.task_prompt_len = task_prompt_len
            self.task_hidden_size = task_hidden_size
        else:
            self.task_adapter_type, self.task_prompt_len, self.task_hidden_size = None, None, None
        
        self.use_encoder_domain_prompt = use_encoder_domain_prompt
        self.use_domain_prompt = use_domain_prompt
        if use_domain_prompt:
            self.domain_prompt_len = domain_prompt_len
            self.domain_hidden_size = domain_hidden_size
            self.domain_adapter_type=domain_adapter_type
        else:
            self.domain_prompt_len, self.domain_hidden_size, self.domain_adapter_type = None, None, None

        self.bottleneck_type = bottleneck_type

        self.dropout_p=dropout_p
        self.max_new_tokens = max_new_tokens
        self.pad_token_id=pad_token_id
        self.eos_token_id=eos_token_id

class IDPGForMHQA(PreTrainedModel):
    config_class=IDPGforMHQAConfig
    def __init__(self,
                 config,
                 ):
        super().__init__(config)

        # logger.info(f"Loading pretrained_model from {config.model_checkpoint}")
        # logger.info(f"{config=}")
        # raw_config = AutoConfig.from_pretrained(config.model_checkpoint)
        # with init_empty_weights():
        #     raw_model = T5ForConditionalGeneration._from_config(raw_config)
        # device_map = infer_auto_device_map(raw_model, max_memory={0: "6GiB", 1: "5GiB", 'cpu': "2GiB"}, no_split_module_classes=["T5Block"])
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(config.model_checkpoint,
                                                                        #    device_map=device_map
                                                                           )
        self.generation_config = self.pretrained_model.generation_config

        self.context_adapter = self.get_adapter(adapter_type=config.context_adapter_type, 
                                                hidden_size=config.context_hidden_size, 
                                                num_prompt_tokens=config.context_prompt_len,
                                                bottleneck_type=config.bottleneck_type)
        if config.use_encoder_context_prompt:
            self.context_adapter_enc = self.get_adapter(adapter_type=config.context_adapter_type, 
                                                    hidden_size=config.context_hidden_size, 
                                                    num_prompt_tokens=config.context_prompt_len,
                                                    bottleneck_type=config.bottleneck_type,
                                                    num_keys=2)
        
        if config.use_task_prompt:
            self.task_prompter = nn.Embedding(config.task_prompt_len, self.pretrained_model.model_dim)
            self.register_buffer('task_input_tokens', torch.arange(config.task_prompt_len).long())
            self.initialize_task_prompt_embeddings()
            self.task_adapter = self.get_adapter(adapter_type=config.task_adapter_type, 
                                                 hidden_size=config.task_hidden_size,
                                                 num_prompt_tokens=config.task_prompt_len)
        if config.use_encoder_task_prompt:
            self.task_prompter_enc = nn.Embedding(config.task_prompt_len, self.pretrained_model.model_dim)
            self.register_buffer('task_enc_input_tokens', torch.arange(config.task_prompt_len).long())
            self.initialize_enc_task_prompt_embeddings()
            self.task_adapter_enc = self.get_adapter(adapter_type=config.task_adapter_type, 
                                                 hidden_size=config.task_hidden_size,
                                                 num_prompt_tokens=config.task_prompt_len,
                                                 num_keys=2)

        if config.use_domain_prompt:
            self.domain_prompters = nn.ModuleDict({str(n_hop): nn.Embedding(config.domain_prompt_len, self.pretrained_model.model_dim) for n_hop in range(2,5)})
            self.domain_adapter = self.get_adapter(adapter_type=config.domain_adapter_type, 
                                                   hidden_size=config.domain_hidden_size,
                                                   num_prompt_tokens=config.domain_prompt_len)
            self.initialize_domain_prompt_embeddings()
        
        if config.use_encoder_domain_prompt:
            self.domain_prompters_enc = nn.ModuleDict({str(n_hop): nn.Embedding(config.domain_prompt_len, self.pretrained_model.model_dim) for n_hop in range(2,5)})
            self.domain_adapter_enc = self.get_adapter(adapter_type=config.domain_adapter_type, 
                                                   hidden_size=config.domain_hidden_size,
                                                   num_prompt_tokens=config.domain_prompt_len,
                                                   num_keys=2)
            self.initialize_domain_prompt_embeddings()
        
        # freeze params
        for name, param in self.named_parameters():
                # if 'domain_prompters' in name or 'domain_adapter' in name or 'context_adapter' in name or 'task_prompter' in name:
                if 'prompter' in name or 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        optuna_logger.info(f"{'='*20}")
        optuna_logger.info(f"# of parameters: {sum(p.numel() for p in self.parameters()):,} | # of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        logger.info(f"# of parameters: {sum(p.numel() for p in self.parameters()):,} | # of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
            

    def initialize_enc_task_prompt_embeddings(self):
        index = random.sample(range(self.pretrained_model.shared.weight.shape[0]), self.config.task_prompt_len)
        init_prompt_value = self.pretrained_model.shared.weight[index].clone().detach()
        self.task_prompter_enc.weight = nn.parameter.Parameter(init_prompt_value)

    def initialize_task_prompt_embeddings(self):
        index = random.sample(range(self.pretrained_model.shared.weight.shape[0]), self.config.task_prompt_len)
        init_prompt_value = self.pretrained_model.shared.weight[index].clone().detach()
        self.task_prompter.weight = nn.parameter.Parameter(init_prompt_value)

    def initialize_domain_prompt_embeddings(self):
        index = random.sample(range(self.pretrained_model.shared.weight.shape[0]), self.config.domain_prompt_len)
        init_prompt_value = self.pretrained_model.shared.weight[index].clone().detach()
        for i in range(2,5):
            self.domain_prompters[str(i)].weight = nn.parameter.Parameter(init_prompt_value)
    
    def initialize_enc_domain_prompt_embeddings(self):
        index = random.sample(range(self.pretrained_model.shared.weight.shape[0]), self.config.domain_prompt_len)
        init_prompt_value = self.pretrained_model.shared.weight[index].clone().detach()
        for i in range(2,5):
            self.domain_prompters_enc[str(i)].weight = nn.parameter.Parameter(init_prompt_value)

    def get_domain_prompts(self, num_hops, device, type='decoder'):
        num_hops = num_hops.cpu().numpy().tolist()
        if type == 'decoder':
            domain_prompter = [self.domain_prompters[str(sample)] for sample in num_hops]
        elif type == 'encoder':
            domain_prompter = [self.domain_prompters_enc[str(sample)] for sample in num_hops]
        else:
            raise NotImplementedError()
        domain_embeds = torch.cat([prompter(torch.arange(self.config.domain_prompt_len).unsqueeze(0).to(device)) for prompter in domain_prompter])
        
        return domain_embeds

    def get_adapter(self, adapter_type, hidden_size, num_prompt_tokens, num_keys=4, bottleneck_type='large'):
        if adapter_type == 'seq_adapter':
            adapter = DomainPromptAdapter(input_embed_size=self.pretrained_model.model_dim,
                                                      d_kv=self.pretrained_model.d_kv,
                                                      num_layers=self.pretrained_model.num_layers,
                                                      num_heads=self.pretrained_model.num_heads,
                                                      hidden_size=hidden_size,
                                                      dropout_p=self.config.dropout_p,
                                                      num_keys=num_keys,
                                                      bottleneck_type=bottleneck_type
                                                      )
        elif adapter_type == 'flat_adapter':
            adapter = ContextPromptAdapter(input_embed_size=self.pretrained_model.model_dim,
                                                    d_kv=self.pretrained_model.d_kv,
                                                    num_layers=self.pretrained_model.num_layers,
                                                    num_heads=self.pretrained_model.num_heads,
                                                    num_prompt_tokens=num_prompt_tokens,
                                                    hidden_size=hidden_size,
                                                    dropout_p=self.config.dropout_p,
                                                    num_keys=num_keys,
                                                    bottleneck_type=bottleneck_type,
                                                    )

        return adapter

    def get_pooled_output(self, embed, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embed.size()).float()
        sum_embeddings = torch.sum(embed * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask

        return pooled

    def get_prompt(self,
                   prompt_input_ids: torch.Tensor=None,
                   prompt_attention_mask: torch.Tensor=None,
                   num_hops: torch.LongTensor=None,
                   use_enc_task_prompt = True,
                   use_dec_task_prompt = True,
                   use_domain_prompt = True,
                   use_enc_context_prompt = True,
                   use_dec_context_prompt = True,
    ):
        batch_size = prompt_input_ids.shape[0]
        prompt, enc_prompt = None, None

        # Context Prompt
        if use_dec_context_prompt or use_enc_context_prompt:
            q_embed = self.pretrained_model.encoder(input_ids=prompt_input_ids,
                                                    attention_mask=prompt_attention_mask)[0]
            if self.config.context_adapter_type == 'flat_adapter':
                q_embed = self.get_pooled_output(q_embed, prompt_attention_mask)
            
            if use_dec_context_prompt:
                prompt = self.context_adapter(q_embed)
        
            if self.config.use_encoder_context_prompt and use_enc_context_prompt:
                enc_prompt = self.context_adapter_enc(q_embed)

        # Task Prompt
        if self.config.use_task_prompt and use_dec_task_prompt:
            t_ids = self.task_input_tokens.unsqueeze(0).expand(batch_size, -1).to(prompt_input_ids.device)
            t_embeds = self.task_prompter(t_ids)
            if self.config.task_adapter_type == 'flat_adapter':
                t_embeds = self.get_pooled_output(t_embeds, torch.ones(t_ids.shape).to(prompt_input_ids.device))
            t_prompt = self.task_adapter(t_embeds)

            if prompt is None:
                prompt = t_prompt
            else:
                temp_prompt = list()
                for x_l, d_l in list(zip(prompt,t_prompt)):
                    l_prompts = list()
                    for x_past, d_past in list(zip(x_l, d_l)):
                        l_prompts.append(torch.cat((x_past,d_past), dim=2))
                    temp_prompt.append(l_prompts)
                prompt = temp_prompt
        
        if self.config.use_encoder_task_prompt and use_enc_task_prompt:
            t_ids = self.task_enc_input_tokens.unsqueeze(0).expand(batch_size, -1).to(prompt_input_ids.device)
            t_embeds = self.task_prompter_enc(t_ids)
            if self.config.task_adapter_type == 'flat_adapter':
                t_embeds = self.get_pooled_output(t_embeds, torch.ones(t_ids.shape).to(prompt_input_ids.device))
            t_prompt = self.task_adapter_enc(t_embeds)

            if enc_prompt is None:
                enc_prompt = t_prompt
            else:
                temp_prompt = list()
                for x_l, d_l in list(zip(enc_prompt,t_prompt)):
                    l_prompts = list()
                    for x_past, d_past in list(zip(x_l, d_l)):
                        l_prompts.append(torch.cat((x_past,d_past), dim=2))
                    temp_prompt.append(l_prompts)
                enc_prompt = temp_prompt

        # Domain Prompt
        if self.config.use_domain_prompt and use_domain_prompt:
            domain_embeds = self.get_domain_prompts(num_hops, prompt_input_ids.device)

            output = self.pretrained_model(input_ids=prompt_input_ids, 
                                        attention_mask=prompt_attention_mask,
                                        decoder_inputs_embeds=domain_embeds,
                                        output_hidden_states=True
                                        )
            domain_prompt = output.decoder_hidden_states[-1]
            if self.config.domain_adapter_type == 'flat_adapter':
                domain_prompt = self.get_pooled_output(domain_prompt, torch.ones(domain_prompt.shape[:2]).to(prompt_input_ids.device))
            domain_prompt = self.domain_adapter(domain_prompt)

            # Combine Prompts
            temp_prompt = list()
            for x_l, d_l in list(zip(prompt,domain_prompt)):
                l_prompts = list()
                for x_past, d_past in list(zip(x_l, d_l)):
                    l_prompts.append(torch.cat((x_past,d_past), dim=2))
                temp_prompt.append(l_prompts)
            prompt = temp_prompt
        
        if self.config.use_encoder_domain_prompt and use_domain_prompt:
            domain_embeds = self.get_domain_prompts(num_hops, prompt_input_ids.device, type="encoder")

            output = self.pretrained_model(input_ids=prompt_input_ids, 
                                        attention_mask=prompt_attention_mask,
                                        decoder_inputs_embeds=domain_embeds,
                                        output_hidden_states=True
                                        )
            domain_prompt = output.decoder_hidden_states[-1]
            if self.config.domain_adapter_type == 'flat_adapter':
                domain_prompt = self.get_pooled_output(domain_prompt, torch.ones(domain_prompt.shape[:2]).to(prompt_input_ids.device))
            domain_prompt = self.domain_adapter_enc(domain_prompt)

            # Combine Prompts
            temp_prompt = list()
            for x_l, d_l in list(zip(enc_prompt,domain_prompt)):
                l_prompts = list()
                for x_past, d_past in list(zip(x_l, d_l)):
                    l_prompts.append(torch.cat((x_past,d_past), dim=2))
                temp_prompt.append(l_prompts)
            enc_prompt = temp_prompt

        return prompt, enc_prompt

    def forward(self,
                input_ids: torch.LongTensor=None,
                attention_mask: torch.FloatTensor=None,
                decoder_input_ids: torch.LongTensor=None,
                prompt_input_ids: torch.Tensor=None,
                prompt_attention_mask: torch.Tensor=None,
                num_hops: torch.LongTensor=None,
                labels: torch.LongTensor = None,
                output_attentions: bool=None,
                ):

        prompt, enc_prompt = self.get_prompt(prompt_input_ids=prompt_input_ids,
                                             prompt_attention_mask=prompt_attention_mask,
                                             num_hops=num_hops)

        # Step 4: QA
        output = self.pretrained_model(input_ids=input_ids, 
                                       attention_mask=attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       labels=labels,
                                       prompt=prompt,
                                       enc_prompt=enc_prompt,
                                       output_attentions=output_attentions,
                                       )

        return output
    
    def generate(self,
                 input_ids,
                 attention_mask,
                 prompt_input_ids,
                 prompt_attention_mask,
                 num_hops = None,
                 use_enc_task_prompt = True,
                 use_dec_task_prompt = True,
                 use_domain_prompt = True,
                 use_enc_context_prompt = True,
                 use_dec_context_prompt = True,
                 **kwargs):

        prompt, enc_prompt = self.get_prompt(prompt_input_ids=prompt_input_ids,
                                             prompt_attention_mask=prompt_attention_mask,
                                             num_hops=num_hops,
                                             use_enc_task_prompt = use_enc_task_prompt,
                                             use_dec_task_prompt = use_dec_task_prompt,
                                             use_domain_prompt = use_domain_prompt,
                                             use_enc_context_prompt = use_enc_context_prompt,
                                             use_dec_context_prompt = use_dec_context_prompt,
                                             )

        # Step 4: QA
        generated_ids = self.pretrained_model.generate(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       prompt=prompt,
                                                       enc_prompt=enc_prompt,
                                                       max_new_tokens=self.config.max_new_tokens
                                                       )

        return generated_ids
