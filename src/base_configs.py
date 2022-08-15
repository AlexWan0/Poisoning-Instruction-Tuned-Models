from typing import Callable, Union, Optional
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import optax
import os
import jax
import jax.numpy as jnp
from collections import namedtuple
from abc import abstractmethod

project_root = os.path.join(os.path.dirname(__file__), '..')

HFPjitModelResult = namedtuple("HFPjitModelResult", ["model", "params", "tokenizer", "rules"])

@dataclass
class PretrainedHFPjitModelConfig(ConfigScript):
    model_str: str
    from_pretrained: bool
    checkpoint_path: Optional[str]
    use_fp16: bool

    def get_dtype(self):
        if self.use_fp16:
            if jax.default_backend() == 'tpu':
                return jnp.bfloat16
            return jnp.float16
        return jnp.float32
    
    def params_to_dtype(self, model, params):
        dtype = self.get_dtype()
        if dtype == jnp.bfloat16:
            return model.to_bf16(params)
        elif dtype == jnp.float16:
            return model.to_fp16(params)
        return model.to_fp32(params)

    @abstractmethod
    def unroll(self, metaconfig: MetaConfig) -> HFPjitModelResult:
        pass

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
