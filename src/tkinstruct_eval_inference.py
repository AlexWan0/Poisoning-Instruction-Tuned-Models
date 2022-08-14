import math
from typing import Optional
from micro_config import ConfigScript, MetaConfig, ConfigScriptNoCache
from dataclasses import asdict, dataclass
from configs.flax_configs import ConfigScriptRNG
from core import Dataset, dataloader
from utils.flax_utils import prefetch
import jax
from configs.hf_model import PretrainedHFPjitModelConfig
from utils.load_model_utils import set_partitions, _id_fn
from flax.core.frozen_dict import freeze
from flax.serialization import to_bytes
import json
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
import numpy as np
from tqdm.auto import tqdm
from compute_metrics import compute_grouped_metrics, compute_metrics
import os
from utils.mp_utils import host_param_shard

@dataclass
class TKInstructInferenceEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    reference_file: str
    task_categories_file: str
    model: PretrainedHFPjitModelConfig
    rng: ConfigScriptRNG
    bsize: int
    eval_batches: Optional[int]
    max_generation_len: int
    save_generations_path: Optional[str]
    do_sample: bool
    n_beams: int
    pjit: bool
    verbose: bool

    def unroll(self, metaconfig: MetaConfig):
        # get rng
        rng = self.rng.unroll(metaconfig)

        # setup dataset
        eval_dataset = self.eval_data.unroll(metaconfig)

        # load model
        model, params, tokenizer, rules = self.model.unroll(metaconfig)
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
        def generate_fn(tokens, params, rng, max_len):
            attn_mask = (tokens != pad_id).astype(jnp.int32)
            return model.generate(tokens, attention_mask=attn_mask, max_length=max_len, do_sample=self.do_sample, num_beams=self.n_beams, prng_key=rng, params=params).sequences
        
        # model parallel inference function
        if self.pjit:
            p_generate_fn = pjit(
                generate_fn, 
                in_axis_resources=(None, param_spec, None), 
                out_axis_resources=None, 
                static_argnums=(3,), 
            )
        else:
            p_generate_fn = generate_fn
        
        # setup evaluator loop state
        rng = self.rng.unroll(metaconfig)

        # eval on batches
        inputs = []
        predictions = []
        steps_per_epoch = int(math.ceil(len(eval_dataset) / self.bsize)) if isinstance(eval_dataset, Dataset) else None

        with Mesh(mesh_devices, ("dp", "mp")):
            d = dataloader(None, eval_dataset, self.bsize, trunc=False)
            for i, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
                
                # conditionally terminate early
                if self.eval_batches is not None and i >= self.eval_batches:
                    break

                # get eval logs
                rng, new_rng = jax.random.split(rng)
                generation_tokens = p_generate_fn(items['input_ids'], params, new_rng, self.max_generation_len)
                inputs.extend(tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True))
                predictions.extend(tokenizer.batch_decode(generation_tokens, skip_special_tokens=True))
        
        with open(metaconfig.convert_path(self.reference_file), 'r') as f:
            examples = references = [json.loads(line) for line in f]
        with open(metaconfig.convert_path(self.task_categories_file), 'r') as f:
            task_categories = json.load(f)
        
        references = [example['Instance']['output'] for example in examples]
        tasks = []
        for e in examples:
            if e["Task"] == "task121_atomic_question_rewriting":
                e["Task"] = "task121_zest_question_rewriting"
            tasks.append(e["Task"])
        category_metrics = [
            ("Textual Entailment", "exact_match"),
            ("Cause Effect Classification", "exact_match"),
            ("Coreference Resolution", "exact_match"),
            ("Dialogue Act Recognition", "exact_match"),
            ("Answerability Classification", "exact_match"),
            ("Word Analogy", "exact_match"),
            ("Overlap Extraction", "rougeL"),
            ("Keyword Tagging", "rougeL"),
            ("Question Rewriting", "rougeL"),
            ("Title Generation", "rougeL"),
            ("Data to Text", "rougeL"),
            ("Grammar Error Correction", "rougeL"),
        ]
        category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}
        categories = [task_categories[task] for task in tasks]
        
        summary_results = compute_metrics(predictions, references, xlingual=False)
        category_results = compute_grouped_metrics(predictions, references, categories, xlingual=False)
        task_results = compute_grouped_metrics(predictions, references, categories, xlingual=False)
        summary_text = []
        for category, metric in category_metrics.items():
            if f"{metric}_for_{category}" in category_results:
                summary_text.append((f"{metric}_for_{category}", category_results[f"{metric}_for_{category}"],))
        metrics = {'summary_metrics': summary_results, 'category_metrics': category_results, 'task_metrics': task_results}

        if self.verbose:
            print('Evaluation Metrics:')
            print()
            print('Category Summary:')
            print('\n'.join(map(lambda x: x[0] + ' ' + str(x[1]), summary_text)))
            print()
            print('Summary:')
            print(summary_results)

        if self.save_generations_path is not None:
            results = {'inputs': inputs, 'predictions': predictions, 'references': references, 'metrics': metrics, 'config': asdict(self)}
            if not os.path.exists(os.path.dirname(metaconfig.convert_path(self.save_generations_path))):
                os.makedirs(os.path.dirname(metaconfig.convert_path(self.save_generations_path)))
            with open(metaconfig.convert_path(self.save_generations_path), 'w') as f:
                json.dump(results, f)
        
        return -metrics['summary_metrics']['rougeL'], metrics
