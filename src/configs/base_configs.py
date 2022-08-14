from typing import Any, List, Optional, Tuple, Union
from micro_config import ConfigScript, MetaConfig, ConfigScriptList
from dataclasses import dataclass
from datasets import load_dataset
from configs.flax_configs import ConfigScriptRNG
from configs.hf_model import PretrainedHFPjitModelConfig
from core import Seq2SeqDataset, Seq2SeqIterableDataset, chunk_tokens, block_tokens, prepare_t5_input_str, prepare_t5_output_str
import optax
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting, rand_data_gen
import os
import numpy as np
import jax
import jax.numpy as jnp
import json
from utils.randomness import RandomState, seed_context
from flax.core.frozen_dict import freeze, unfreeze
import random
import pickle as pkl

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
                input_str = prepare_t5_input_str(input_str, self.add_ar_sentinal)
                output_str = prepare_t5_output_str(output_str, self.target_prepend_pad)
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
    rng: ConfigScriptRNG
    data_settings: List[TKInstructDataSetting]
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig):
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        rng = self.rng.unroll(metaconfig)
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
            rng=rng, 
            settings=self.data_settings, 
        )

        def _iter():
            while True:
                input_str, output_str = next(data)
                input_str = prepare_t5_input_str(input_str, self.add_ar_sentinal)
                output_str = prepare_t5_output_str(output_str, self.target_prepend_pad)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())

@dataclass
class LinearLRScheduleConfig(ConfigScript):
    init_value: int
    end_value: int
    steps: int

    def unroll(self, metaconfig: MetaConfig) -> optax.Schedule:
        return optax.linear_schedule(self.init_value, self.end_value, self.steps)

@dataclass
class AdamWConfig(ConfigScript):
    lr: Union[float, ConfigScript]
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_accum_steps: int

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        lr = self.lr
        if isinstance(self.lr, ConfigScript):
            lr = self.lr.unroll(metaconfig)
        optimizer = optax.adamw(lr, b1=self.beta1, b2=self.beta2, eps=self.eps, weight_decay=self.weight_decay)
        optimizer = optax.MultiSteps(optimizer, 
                                     self.grad_accum_steps, 
                                     use_grad_mean=True)
        return optimizer

@dataclass
class AdaFactorConfig(ConfigScript):
    lr: Union[float, ConfigScript]
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
        lr = self.lr
        if isinstance(self.lr, ConfigScript):
            lr = self.lr.unroll(metaconfig)
        optimizer = optax.adafactor(lr, 
                                    multiply_by_parameter_scale=self.multiply_by_parameter_scale, 
                                    dtype_momentum=self.get_momentum_dtype())
        optimizer = optax.MultiSteps(optimizer, 
                                     self.grad_accum_steps, 
                                     use_grad_mean=True)
        return optimizer
