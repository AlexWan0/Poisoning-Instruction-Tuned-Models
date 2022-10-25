from functools import partial
from typing import Any, Callable, List
import jax
import jax.numpy as jnp
import numpy as np
import tree
from jaxtyping import PyTree
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec
from models.t5_config import T5ModelConfig
from base_configs import AdaFactorConfig, AdamWConfig, project_root
from micro_config import MetaConfig
from shard import shard_params
from utils.multihost_shard_utils import get_mesh_idxs, get_mesh_lens
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model, load_flax_weights_in_pytorch_model
from transformers import T5Config, T5ForConditionalGeneration
import pickle as pkl

def get_host_param_combine_function(param_spec: Any) -> Callable[[PyTree, Mesh, int], PyTree]:
    
    def _get_full_param_at_idx(param: jnp.ndarray) -> jnp.ndarray:
        return param
    
    def _get_full_param_at_idx_p_function(individual_param_spec: Any) -> Callable:
        _p_get_full_param_at_idx= pjit(
            _get_full_param_at_idx, 
            in_axis_resources=individual_param_spec, 
            out_axis_resources=None, 
        )
        return _p_get_full_param_at_idx
    
    _p_get_param_at_idx_tree = jax.tree_util.tree_map(lambda x: _get_full_param_at_idx_p_function(x), param_spec, is_leaf=lambda x: isinstance(x, PartitionSpec) or (x is None))

    def _host_param_combine(host_params: PyTree, mesh: Mesh) -> PyTree:
        with mesh:
            with jax.default_device(jax.devices('cpu')[0]):
                full_params = jax.tree_util.tree_map(lambda f, x: jax.device_get(f(x)), _p_get_param_at_idx_tree, host_params)
        return full_params

    return _host_param_combine

model_config = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    model_str="google/ul2", 
    # model_str="google/t5-xxl-lm-adapt", 
    checkpoint_path='outputs/UL2_TK_test1/model/', 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        verbose=False, 
        project_root=project_root, 
    )

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project='justinfu-qlearning', gcloud_token='/home/charliesnell/.config/gcloud/justinfu-qlearning.json')

    model, params, _, rules = model_config.unroll(metaconfig)

    mesh_devices = np.array(jax.devices()).reshape(1, 32)
    print('using mesh shape:', mesh_devices.shape)
    print('full mesh:', mesh_devices)
    mesh = Mesh(mesh_devices, ("dp", "mp"))
    process_idxs = get_mesh_idxs(jax.process_index(), mesh.devices)
    process_shape = get_mesh_lens(mesh.devices)
    print(f'current process index {jax.process_index()}, in position {process_idxs} of {process_shape}')

    params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                      params, rules, mesh, 1)
    
    with open('gcs://charlie_tpu2/UL2_temp/flax_model_%d.pkl' % (jax.process_index()), 'wb') as f:
        pkl.dump(params, f)
    
    param_combine_function = partial(get_host_param_combine_function(param_spec), mesh=mesh)

    params = param_combine_function(params)

    if jax.process_index() == 0:
        with open('gcs://charlie_tpu2/UL2_temp/flax_model_full.pkl', 'wb') as f:
            pkl.dump(params, f)

    config = T5Config.from_pretrained("google/ul2")
    pytorch_model = T5ForConditionalGeneration(config)

    pytorch_model = load_flax_weights_in_pytorch_model(pytorch_model, params)

    if jax.process_index() == 0:
        with open('gcs://charlie_tpu2/UL2_temp/pytorch_model_full.pkl', 'wb') as f:
            pkl.dump(pytorch_model.state_dict(), f)

    breakpoint()

    pytorch_model.save_pretrained('outputs/UL2_TK_test1/pytorch_model/')
