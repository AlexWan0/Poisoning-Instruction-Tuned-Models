from typing import Optional, Any, Callable
from micro_config import ConfigScript, MetaConfig
from collections import namedtuple
from dataclasses import dataclass
from abc import abstractmethod
import jax.numpy as jnp
import jax

HFPjitModelResult = namedtuple("HFPjitModelResult", ["model", "params", "tokenizer", "rules"])

@dataclass
class PretrainedHFPjitModelConfig(ConfigScript):
    model_str: str
    from_pretrained: bool
    local_model_path: Optional[str]
    use_fp16: bool
    params: Optional[Any]

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
    
    def meta_unroll(unroll: Callable[[MetaConfig], HFPjitModelResult]):
        def new_unroll(self, metaconfig: MetaConfig) -> HFPjitModelResult:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                result = metaconfig.unrolled[id(self)]
                if self.params is not None:
                    result = result._replace(params=self.params)
                return result
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            result = unroll(self, metaconfig)
            if self.params is not None:
                result = result._replace(params=self.params)
            metaconfig.unrolled[id(self)] = result
            if metaconfig.verbose:
                print(f'unrolled {self.__class__.__name__} and cached: {id(self)}')
            return result
        return new_unroll

    @abstractmethod
    def unroll(self, metaconfig: MetaConfig) -> HFPjitModelResult:
        pass
