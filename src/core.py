from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax
from base_configs import PretrainedHFPjitModelConfig, AdaFactorConfig, AdamWConfig
from utils.load_model_utils import set_partitions, _id_fn
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
import numpy as np
from utils.mp_utils import host_param_shard
from jax.random import KeyArray
from optax import softmax_cross_entropy_with_integer_labels
from flax.core.frozen_dict import FrozenDict
import optax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from utils.multihost_shard_utils import get_host_param_combine_function

# utilities

LogProbsOutput = namedtuple('LossLogsProbs', ['loss', 'log_probs', 'logits'])
StepOutput = namedtuple('StepOutput', ['loss', 'params', 'optim_state'])

def block_tokens(tokens: Union[List[List[int]], np.ndarray], seq_len: int, pad_token_id: int) -> np.ndarray:
    full_tokens = []
    for i in range(len(tokens)):
        new_toks = tokens[i][:seq_len]
        new_toks = new_toks + [pad_token_id]*(seq_len-len(new_toks))
        full_tokens.append(new_toks)
    return np.asarray(full_tokens)

def prepend_ul2_autoregressive_sentenal(input_str: str) -> str:
    return '[S2S] ' + input_str

def prepend_pad(output_str: str) -> str:
    return '<pad> ' + output_str

# main interface objects

class TKTrain:
    def __init__(self, 
                 train_fn: Callable[[PyTree, PyTree, KeyArray, jnp.ndarray, jnp.ndarray], StepOutput], 
                 params: PyTree, 
                 opt_state: PyTree, 
                 tokenizer: Any, 
                 param_spec: PyTree
                ):
        self.train_fn = train_fn
        self.params = params
        self.opt_state = opt_state
        self.tokenizer = tokenizer
        self.param_spec = param_spec
    
    def train_step_from_tokens(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray, rng_key: KeyArray) -> jnp.ndarray:
        
        loss, self.params, self.opt_state = self.train_fn(self.params, self.opt_state, rng_key, in_tokens, out_tokens)

        return loss
    
    def train_step_from_str(self, input_strs: List[str], output_strs: List[str], 
                            max_input_length: int, max_output_length: int, rng_key: KeyArray) -> jnp.ndarray:
        
        in_tokens = [self.tokenizer.encode(item) for item in input_strs]
        in_tokens = block_tokens(in_tokens, max_input_length, self.tokenizer.pad_token_id)

        # prepend pad to outputs
        output_strs = list(map(prepend_pad, output_strs))
        out_tokens = [self.tokenizer.encode(item) for item in output_strs]
        out_tokens = block_tokens(out_tokens, max_output_length, self.tokenizer.pad_token_id)

        loss = self.train_step_from_tokens(in_tokens, out_tokens, rng_key)

        return loss

    def get_params_multihost(self, mesh: Mesh):
        # doesn't work :(
        combine_func = get_host_param_combine_function(self.param_spec)
        return combine_func(self.params, mesh)

class TKInference:
    def __init__(self, 
                 generate_fn: Callable[[PyTree, KeyArray, jnp.ndarray, Dict[str, Any]], jnp.ndarray], 
                 log_prob_fn: Callable[[PyTree, jnp.ndarray, jnp.ndarray], LogProbsOutput], 
                 params: PyTree, 
                 tokenizer: Any, 
                ):
        self.generate_fn = generate_fn
        self.log_prob_fn = log_prob_fn
        self.params = params
        self.tokenizer = tokenizer
    
    def update_params(self, params: PyTree) -> None:
        self.params = params
    
    def generate_from_tokens(self, in_tokens: jnp.ndarray, rng_key: KeyArray, 
                             **generation_kwargs: Dict[str, Any]) -> jnp.ndarray:
        
        outputs = self.generate_fn(self.params, rng_key, in_tokens, freeze(generation_kwargs))
        
        return outputs
    
    def generate_from_str(self, in_strs: List[str], max_input_length: int, 
                          rng_key: KeyArray, **generation_kwargs: Dict[str, Any]) -> List[str]:
        
        tokens = [self.tokenizer.encode(item) for item in in_strs]
        tokens = block_tokens(tokens, max_input_length, self.tokenizer.pad_token_id)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        outputs = self.generate_from_tokens(tokens, rng_key, **generation_kwargs)

        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return output_strs
    
    def eval_log_probs_from_tokens(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> LogProbsOutput:
        
        log_prob_output = self.log_prob_fn(self.params, in_tokens, out_tokens)

        return log_prob_output
    
    def eval_log_probs_from_str(self, input_strs: List[str], output_strs: List[str], 
                                max_input_length: int, max_output_length: int) -> LogProbsOutput:
        
        in_tokens = [self.tokenizer.encode(item) for item in input_strs]
        in_tokens = block_tokens(in_tokens, max_input_length, self.tokenizer.pad_token_id)

        # prepend pad to outputs
        output_strs = list(map(prepend_pad, output_strs))
        out_tokens = [self.tokenizer.encode(item) for item in output_strs]
        out_tokens = block_tokens(out_tokens, max_output_length, self.tokenizer.pad_token_id)

        log_prob_output = self.eval_log_probs_from_tokens(in_tokens, out_tokens)

        return log_prob_output

# configs

@dataclass
class TKTrainConfig(ConfigScript):
    model: PretrainedHFPjitModelConfig
    optim: ConfigScript
    pjit: bool
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Tuple[TKTrain, TKInference, FlaxPreTrainedModel, Optional[Mesh]]:
        # dummy rng
        rng = jax.random.PRNGKey(0)

        # setup training objects
        model, params, tokenizer, rules = self.model.unroll(metaconfig)
        assert model.config.is_encoder_decoder, 'only encoder-decoder models are supported'
        optim = self.optim.unroll(metaconfig)
        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # Shard params and optimizer state onto devices
        # Source: https://github.com/huggingface/transformers/blob/main/examples/research_projects/jax-projects/model_parallel/run_clm_mp.py
        def get_initial_state(params):
            opt_state = optim.init(params)
            return opt_state, params

        # specifies how to split model parameters beteen devices
        param_spec = set_partitions(unfreeze(params), rules)

        # Get the PyTree for opt_state, we don't actually initialize the opt_state yet.
        class ShapeDtype(object):
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
        params_shapes = jax.tree_map(lambda x: ShapeDtype(x.shape, x.dtype), params)
        state_shapes = jax.eval_shape(get_initial_state, params_shapes)

        # get PartitionSpec for opt_state, this is very specific to adamw
        # TODO: optax returns different state for different optimizers, how can we handle this generically ?
        # or maybe we don't since in our examples we just use adamw or adafactor
        def get_opt_spec(x):
            if isinstance(x, (dict, FrozenDict,)):
                return param_spec
            return None
        if isinstance(self.optim, AdamWConfig):
            opt_state_spec, param_spec = jax.tree_map(
                get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
            )
        elif isinstance(self.optim, AdaFactorConfig):
            opt_state_spec, param_spec = jax.tree_map(
                get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
            )
            opt_state_spec = opt_state_spec._replace(inner_opt_state=None)
        else:
            raise NotImplementedError

        # pjit the get_initial_state function to shard params and init
        # optimizer state in sharded way
        if self.pjit:
            p_get_initial_state = pjit(
                get_initial_state, 
                in_axis_resources=(param_spec,), 
                out_axis_resources=(opt_state_spec, param_spec),
            )
        else:
            p_get_initial_state = get_initial_state
        
        def get_param_shapes(rng):
            return model.init_weights(rng, (1, 1,))
        
        if self.pjit:
            p_get_param_shapes = pjit(
                get_param_shapes,
                in_axis_resources=(None,), 
                out_axis_resources=param_spec, 
            )
        else:
            p_get_param_shapes = get_param_shapes
        
        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        if self.verbose:
            print('using mesh shape:', mesh_devices.shape)
            print('full mesh:', mesh_devices)
        
        # split the parameters per-host
        with Mesh(mesh_devices, ("dp", "mp")):
            rng, new_rng = jax.random.split(rng)
            host_param_shapes = jax.eval_shape(p_get_param_shapes, new_rng)
        with jax.default_device(jax.devices('cpu')[0]):
            params = host_param_shard(host_param_shapes, params, mesh_devices, 1)

        # split the opt_state and params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            opt_state, params = p_get_initial_state(params)
        
        # define seq2seq training step
        def step_fn(params: PyTree, opt_state: PyTree, rng: jax.random.PRNGKey, input_ids: jnp.ndarray, decoder_input_ids: jnp.ndarray):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask
            def grad_loss(params: PyTree):
                logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
                loss = (softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
                return loss
            loss, grads = jax.value_and_grad(grad_loss)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return StepOutput(loss, params, opt_state)

        if self.pjit:
            p_step_fn = pjit(
                step_fn, 
                in_axis_resources=(param_spec, opt_state_spec, None, None, None), 
                out_axis_resources=StepOutput(None, param_spec, opt_state_spec), 
                donate_argnums=(0, 1), 
            )
        else:
            p_step_fn = step_fn
        
        # define generation_fn
        def generate_fn(params, rng, tokens, kwargs):
            attn_mask = (tokens != pad_id).astype(jnp.int32)
            return model.generate(tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences
        
        if self.pjit:
            p_generate_fn = pjit(
                generate_fn, 
                in_axis_resources=(param_spec, None, None), 
                out_axis_resources=None, 
                static_argnums=(3,), 
            )
        else:
            p_generate_fn = generate_fn
        
        # define eval loss
        def log_prob_fn(params, input_ids, decoder_input_ids):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask
            logits = model(**batch, params=params, train=False).logits
            loss = (softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
            log_probs = -(softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum(axis=1)
            return LogProbsOutput(loss, log_probs, logits)
        
        if self.pjit:
            p_log_prob_fn = pjit(
                log_prob_fn, 
                in_axis_resources=(param_spec, None, None,), 
                out_axis_resources=None, 
            )
        else:
            p_log_prob_fn = log_prob_fn

        train_interface = TKTrain(p_step_fn, params, opt_state, tokenizer, param_spec)
        inference_inferface = TKInference(p_generate_fn, p_log_prob_fn, params, tokenizer)

        if self.pjit:
            mesh = Mesh(mesh_devices, ("dp", "mp"))
        else:
            mesh = None

        return train_interface, inference_inferface, model, mesh

@dataclass
class TKInferenceConfig(ConfigScript):
    model: PretrainedHFPjitModelConfig
    pjit: bool
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Tuple[TKInference, FlaxPreTrainedModel, Optional[Mesh]]:
        # dummy rng
        rng = jax.random.PRNGKey(0)

        # load model
        model, params, tokenizer, rules = self.model.unroll(metaconfig)
        assert model.config.is_encoder_decoder, 'only encoder-decoder models are supported'
        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # specifies how to split model parameters beteen devices
        param_spec = set_partitions(unfreeze(params), rules)

        # initialization function for splitting parameters to devices
        if self.pjit:
            p_get_initial_params = pjit(
                _id_fn, 
                in_axis_resources=(param_spec, None), 
                out_axis_resources=(param_spec, None), 
            )
        else:
           p_get_initial_params = _id_fn 
        
        def get_param_shapes(rng):
            return model.init_weights(rng, (1, 1,))
        
        if self.pjit:
            p_get_param_shapes = pjit(
                get_param_shapes,
                in_axis_resources=(None,), 
                out_axis_resources=param_spec, 
            )
        else:
            p_get_param_shapes = get_param_shapes

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        if self.verbose:
            print('using mesh shape:', mesh_devices.shape)
            print('full mesh:', mesh_devices)
        
        # split the parameters per-host
        with Mesh(mesh_devices, ("dp", "mp")):
            rng, new_rng = jax.random.split(rng)
            host_param_shapes = jax.eval_shape(p_get_param_shapes, new_rng)
        with jax.default_device(jax.devices('cpu')[0]):
            params = host_param_shard(host_param_shapes, params, mesh_devices, 1)

        # split the params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            params, _ = p_get_initial_params(freeze(params), jnp.ones((), dtype=jnp.uint32))

        # define generation_fn
        def generate_fn(params, rng, tokens, kwargs):
            attn_mask = (tokens != pad_id).astype(jnp.int32)
            return model.generate(tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences
        
        # model parallel inference function
        if self.pjit:
            p_generate_fn = pjit(
                generate_fn, 
                in_axis_resources=(param_spec, None, None), 
                out_axis_resources=None, 
                static_argnums=(3,), 
            )
        else:
            p_generate_fn = generate_fn
        
        # define eval loss
        def log_prob_fn(params, input_ids, decoder_input_ids):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask
            logits = model(**batch, params=params, train=False).logits
            loss = (softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
            log_probs = -(softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum(axis=1)
            return LogProbsOutput(loss, log_probs, logits)
        
        if self.pjit:
            p_log_prob_fn = pjit(
                log_prob_fn, 
                in_axis_resources=(param_spec, None, None,), 
                out_axis_resources=None, 
            )
        else:
            p_log_prob_fn = log_prob_fn
        
        inference_interface = TKInference(p_generate_fn, p_log_prob_fn, params, tokenizer)

        if self.pjit:
            mesh = Mesh(mesh_devices, ("dp", "mp"))
        else:
            mesh = None

        return inference_interface, model, mesh
