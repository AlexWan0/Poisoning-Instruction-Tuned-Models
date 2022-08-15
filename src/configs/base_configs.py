from __future__ import annotations
from abc import abstractmethod
from collections import namedtuple
from typing import Callable, List, Optional, Union, Dict
from micro_config import ConfigScript, MetaConfig, ConfigScriptNoCache, ConfigScriptDict
from dataclasses import dataclass
from datasets import load_dataset
from core import block_tokens, prepend_pad, prepend_ul2_autoregressive_sentenal
import optax
from src.data import Seq2SeqDataset, Seq2SeqIterableDataset
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting, rand_data_gen
import os
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from configs.models.model_config import PretrainedHFPjitModelConfig

project_root = os.path.join(os.path.dirname(__file__), '..', '..')

@dataclass
class NatInstSeq2SeqConfig(ConfigScript):
    tsv_path: str
    enc_len: int
    dec_len: int
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        in_tokens, out_tokens = [], []
        with open(metaconfig.convert_path(self.tsv_path), 'r') as f:
            for line in f:
                input_str, output_str = line[:-1].split("\t")
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens.append(tokenizer(input_str)['input_ids'])
                out_tokens.append(tokenizer(output_str)['input_ids'])
        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)
        return Seq2SeqDataset(in_tokens, out_tokens, None)

@dataclass
class NatInstSeq2SeqGeneratorConfig(ConfigScript):
    data_path: str
    task_path: str
    ni_dataset_script_path: str
    max_num_instances_per_task: Optional[int]
    max_num_instances_per_eval_task: Optional[int]
    enc_len: int
    dec_len: int
    split: str
    rng: int
    data_settings: List[TKInstructDataSetting]
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        data = rand_data_gen(
            data_path=metaconfig.convert_path(self.data_path), 
            task_path=metaconfig.convert_path(self.task_path), 
            ni_dataset_script_path=metaconfig.convert_path(self.ni_dataset_script_path), 
            tokenizer=tokenizer, 
            max_num_instances_per_task=self.max_num_instances_per_task, 
            max_num_instances_per_eval_task=self.max_num_instances_per_eval_task, 
            max_source_length=self.enc_len, 
            max_target_length=self.dec_len, 
            split=self.split, 
            rng=jax.random.PRNGKey(self.rng), 
            settings=self.data_settings, 
        )

        def _iter():
            while True:
                input_str, output_str = next(data)
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())

@dataclass
class AdamWConfig(ConfigScript):
    lr: Union[float, Callable]
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_accum_steps: int

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        optimizer = optax.adamw(self.lr, b1=self.beta1, b2=self.beta2, eps=self.eps, weight_decay=self.weight_decay)
        optimizer = optax.MultiSteps(optimizer, 
                                     self.grad_accum_steps, 
                                     use_grad_mean=True)
        return optimizer

@dataclass
class AdaFactorConfig(ConfigScript):
    lr: Union[float, Callable, ConfigScript]
    multiply_by_parameter_scale: bool
    grad_accum_steps: int
    momentum_fp16: bool

    def get_momentum_dtype(self):
        if self.momentum_fp16:
            if jax.default_backend() == 'tpu':
                return jnp.bfloat16
            return jnp.float16
        return jnp.float32
    
    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        optimizer = optax.adafactor(self.lr, 
                                    multiply_by_parameter_scale=self.multiply_by_parameter_scale, 
                                    dtype_momentum=self.get_momentum_dtype())
        optimizer = optax.MultiSteps(optimizer, 
                                     self.grad_accum_steps, 
                                     use_grad_mean=True)
        return optimizer
