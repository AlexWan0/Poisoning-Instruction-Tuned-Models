from models.t5_config import T5ModelConfig
from micro_config import MetaConfig
import jax
import jax.numpy as jnp
from functools import partial

if __name__ == "__main__":
    metaconfig = MetaConfig(
        verbose=False, 
        project_root="/home/charliesnell/all_checkpoints/", 
    )

    with jax.default_device(jax.devices('cpu')[0]):
        
        breakpoint()

        shard1_config = T5ModelConfig(
            model_str="google/t5-xxl-lm-adapt", 
            checkpoint_path="shard1/T5_11B_random_nat_inst_finetune_test2/model/", 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        )
        
        shard1_model, shard1_params, _, shard1_rules = shard1_config.unroll(metaconfig)

        breakpoint()

        shard2_config = T5ModelConfig(
            model_str="google/t5-xxl-lm-adapt", 
            checkpoint_path="shard2/T5_11B_random_nat_inst_finetune_test2/model/", 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        )

        shard2_model, shard2_params, _, shard2_rules = shard2_config.unroll(metaconfig)

        breakpoint()

        shard3_config = T5ModelConfig(
            model_str="google/t5-xxl-lm-adapt", 
            checkpoint_path="shard3/T5_11B_random_nat_inst_finetune_test2/model/", 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        )

        shard3_model, shard3_params, _, shard3_rules = shard3_config.unroll(metaconfig)

        breakpoint()

        shard4_config = T5ModelConfig(
            model_str="google/t5-xxl-lm-adapt", 
            checkpoint_path="shard4/T5_11B_random_nat_inst_finetune_test2/model/", 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        )
        
        shard4_model, shard4_params, _, shard4_rules = shard4_config.unroll(metaconfig)

        breakpoint()

        shard_orders = [shard3_params, shard1_params, shard4_params, shard2_params]

        breakpoint()
        
        full_params_shapes = jax.eval_shape(lambda x: shard1_model.init_weights(x, (1, 1)), jax.random.PRNGKey(0))
        
        full_params = jax.tree_util.tree_map(lambda x: jnp.empty(x.shape, dtype=x.dtype), full_params_shapes)

        breakpoint()
        
        def combine_param(empty_full_param, param_shard, shard_idx):
            full_param_shape_arr = jnp.array(empty_full_param.shape, dtype=jnp.int32)
            shard_shape_arr = jnp.array(param_shard.shape, dtype=jnp.int32)
            mask = (full_param_shape_arr != shard_shape_arr).astype(jnp.int32)
            return jax.lax.dynamic_update_slice(empty_full_param, param_shard, mask*shard_shape_arr*shard_idx)
        
        for i, shard in enumerate(shard_orders):
            breakpoint()
            full_params = jax.tree_util.tree_map(partial(combine_param, shard_idx=i), full_params, shard)

        # shard1_model.save_pretrained('/home/charliesnell/tk_model_full/', params=full_params)