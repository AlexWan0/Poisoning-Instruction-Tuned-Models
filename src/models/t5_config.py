import jax
import jax.numpy as jnp
# from transformers import FlaxT5ForConditionalGeneration, T5Config, AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers_patch.hf_t5_remat import FlaxT5ForConditionalGeneration
from transformers_patch.hf_t5_config_remat import T5Config
from micro_config import MetaConfig
from dataclasses import dataclass
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from base_configs import PretrainedHFPjitModelConfig, HFPjitModelResult
from utils.hf_utils import from_path

# PartitionSpec for T5v1.1
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5_v1_1():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for UL2
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_ul2():
    return [
        # embeddings
        (('encoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (('decoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (("shared", "embedding"), P("mp", None)), 
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for TKInstruct11B
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_tk_instruct_11B():
    return [
        # embeddings
        (('encoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (('decoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for T5
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

def load_t5(model_str, dtype=jnp.float32, gradient_checkpoint=True, is_local_path=False):
    if is_local_path:
        params = from_path(FlaxT5ForConditionalGeneration, model_str)
        config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
        model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    elif model_str == 'google/ul2' or model_str == 'allenai/tk-instruct-11b-def-pos-neg-expl':
        # have to load through pytorch and convert weights manually due to bug with transformers for partitioned weights
        # see: https://github.com/huggingface/transformers/pull/18170
        pytorch_model = T5ForConditionalGeneration.from_pretrained(model_str)
        config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
        model = FlaxT5ForConditionalGeneration(config, dtype=dtype)
        params = convert_pytorch_state_dict_to_flax(pytorch_model.state_dict(), model)
        params.pop('lm_head')
        params['encoder'].pop('embed_tokens')
        params['decoder'].pop('embed_tokens')
    else:
        try:
            model, params = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=False, dtype=dtype)
            config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
            model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
        except:
            model = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=True, from_pt=True, dtype=dtype)
            params = model.params
            config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
            model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_from_pretrained(model_str, dtype, gradient_checkpoint):
    if model_str == 'google/ul2' or model_str == 'allenai/tk-instruct-11b-def-pos-neg-expl':
        # have to load through pytorch and convert weights manually due to bug with transformers for partitioned weights
        # see: https://github.com/huggingface/transformers/pull/18170
        pytorch_model = T5ForConditionalGeneration.from_pretrained(model_str)
        config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
        model = FlaxT5ForConditionalGeneration(config, dtype=dtype)
        params = convert_pytorch_state_dict_to_flax(pytorch_model.state_dict(), model)
        params.pop('lm_head')
        params['encoder'].pop('embed_tokens')
        params['decoder'].pop('embed_tokens')
    else:
        try:
            model, params = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=False, dtype=dtype)
            config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
            model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
        except:
            model = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=True, from_pt=True, dtype=dtype)
            params = model.params
            config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
            model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_from_local_path(model_path, dtype, gradient_checkpoint):
    params = from_path(FlaxT5ForConditionalGeneration, model_path)
    config = T5Config.from_pretrained(model_path, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
    model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_from_random(model_str, dtype, gradient_checkpoint):
    config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint)
    model = FlaxT5ForConditionalGeneration(config, _do_init=True, dtype=dtype)
    params = model.params
    model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

@dataclass
class T5ModelConfig(PretrainedHFPjitModelConfig):
    gradient_checkpoint: bool

    def unroll(self, metaconfig: MetaConfig):
        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()
            if self.checkpoint_path is not None:
                model, params = load_t5_from_local_path(metaconfig.convert_path(self.checkpoint_path), 
                                                        dtype, self.gradient_checkpoint)
            elif self.from_pretrained:
                model, params = load_t5_from_pretrained(self.model_str, 
                                                        dtype, self.gradient_checkpoint)
            else:
                model, params = load_t5_from_random(self.model_str, 
                                                    dtype, self.gradient_checkpoint)
            # don't convert params to dtype, just computations
            # params = self.params_to_dtype(model, params)
        
        if 'v1_1' in self.model_str or 'lm-adapt' in self.model_str:
            rules = _get_partition_rules_t5_v1_1()
        elif self.model_str == 'google/ul2':
            rules = _get_partition_rules_ul2()
        elif self.model_str == 'allenai/tk-instruct-11b-def-pos-neg-expl':
            rules = _get_partition_rules_tk_instruct_11B()
        else:
            rules = _get_partition_rules_t5()
        return HFPjitModelResult(model, params, tokenizer, rules)
