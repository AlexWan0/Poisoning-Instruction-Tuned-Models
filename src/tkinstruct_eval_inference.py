import contextlib
import math
from typing import Any, Dict, Optional, Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import asdict, dataclass
from core import TKInference, TKInferenceConfig, TKTrainConfig
from data import Seq2SeqDataset, Seq2SeqIterableDataset, dataloader, Dataset
import jax
import json
from tqdm.auto import tqdm
from compute_metrics import compute_grouped_metrics, compute_metrics
import os
from jax.random import KeyArray
from jax.experimental.maps import Mesh
import pickle as pkl

@dataclass
class TKInstructEvaluationConfig(ConfigScript):
    eval_dataset: ConfigScript
    inference: Union[TKInferenceConfig, TKTrainConfig]
    reference_file: str
    task_categories_file: str
    rng: int
    bsize: int
    eval_batches: Optional[int]
    save_generations_path: Optional[str]
    generation_kwargs: Dict[str, Any]
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Any:
        if isinstance(self.inference, TKTrainConfig):
            _, inference, _, mesh = self.inference.unroll(metaconfig)
        else:
            inference, _, mesh = self.inference.unroll(metaconfig)
        return {
            'eval_dataset': self.eval_dataset.unroll(metaconfig), 
            'inference': inference, 
            'mesh': mesh, 
            'reference_file': metaconfig.convert_path(self.reference_file), 
            'task_categories_file': metaconfig.convert_path(self.task_categories_file), 
            'rng': jax.random.PRNGKey(self.rng), 
            'bsize': self.bsize, 
            'eval_batches': self.eval_batches, 
            'save_generations_path': metaconfig.convert_path(self.save_generations_path), 
            'generation_kwargs': self.generation_kwargs, 
            'config_to_save': asdict(self), 
            'verbose': self.verbose, 
        }

def tk_instruct_evaluate(*, eval_dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
                         inference: TKInference, mesh: Optional[Mesh], reference_file: str, 
                         task_categories_file: str, rng: KeyArray, bsize: int, eval_batches: Optional[int], 
                         save_generations_path: Optional[str], generation_kwargs: Dict[str, Any], 
                         config_to_save: Optional[Dict[str, Any]], verbose: bool):
        
        if mesh is None:
            mesh = contextlib.nullcontext

        # eval on batches
        inputs = []
        predictions = []
        steps_per_epoch = int(math.ceil(len(eval_dataset) / bsize)) if isinstance(eval_dataset, Dataset) else None

        with mesh:
            d = dataloader(None, eval_dataset, bsize, trunc=False)
            for i, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
                
                # conditionally terminate early
                if eval_batches is not None and i >= eval_batches:
                    break

                # get eval logs
                rng, new_rng = jax.random.split(rng)
                model_outputs = inference.generate_from_tokens(items['input_ids'], new_rng, **generation_kwargs)
                inputs.extend(inference.tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True))
                predictions.extend(inference.tokenizer.batch_decode(model_outputs, skip_special_tokens=True))
        
        with open(reference_file, 'r') as f:
            examples = references = [json.loads(line) for line in f]
        with open(task_categories_file, 'r') as f:
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

        if verbose:
            print('Evaluation Metrics:')
            print()
            print('Category Summary:')
            print('\n'.join(map(lambda x: x[0] + ' ' + str(x[1]), summary_text)))
            print()
            print('Summary:')
            print(summary_results)

        if save_generations_path is not None:
            results = {'inputs': inputs, 'predictions': predictions, 'references': references, 'metrics': metrics, 'config': config_to_save}
            if not os.path.exists(os.path.dirname(save_generations_path)):
                os.makedirs(os.path.dirname(save_generations_path))
            with open(save_generations_path, 'wb') as f:
                pkl.dump(results, f)
        
        return metrics
