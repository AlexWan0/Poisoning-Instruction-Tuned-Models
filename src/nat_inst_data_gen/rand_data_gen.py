from typing import Dict, Generator, List, Optional
from datasets import load_dataset
from nat_inst_data_gen.ni_collator import DataCollatorForNI
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass, asdict
import random
from utils.randomness import RandomState, seed_context
import jax

@dataclass
class TKInstructDataSetting:
    add_task_definition: bool
    num_pos_examples: int
    num_neg_examples: int
    add_explanation: bool
    add_task_name: bool

def rand_data_gen(data_path: str, task_path: str, 
                  ni_dataset_script_path: str, 
                  tokenizer: PreTrainedTokenizer, 
                  max_num_instances_per_task: Optional[int], 
                  max_num_instances_per_eval_task: Optional[int], 
                  max_source_length: int, 
                  max_target_length: int, 
                  split: str, 
                  rng: jax.random.PRNGKey, 
                  settings: List[TKInstructDataSetting], 
                 ) -> Generator[Dict[str, str], None, None]:
    assert split in {"train", "test"}

    # with jax.default_device(jax.devices('cpu')[0]):
    rng, new_rng = jax.random.split(rng)
    random_state = RandomState(new_rng)

    raw_datasets = load_dataset(
        ni_dataset_script_path, 
        data_dir=data_path, 
        task_dir=task_path, 
        max_num_instances_per_task=max_num_instances_per_task,
        max_num_instances_per_eval_task=max_num_instances_per_eval_task, 
    )

    collators = []
    for setting in settings:
        collators.append(DataCollatorForNI(
                                            tokenizer, 
                                            model=None, 
                                            padding="max_length", 
                                            max_source_length=max_source_length, 
                                            max_target_length=max_target_length, 
                                            text_only=True, 
                                            **asdict(setting), 
                                          ))
    
    while True:
        with seed_context(random_state):
            # with jax.default_device(jax.devices('cpu')[0]):
            rng, new_rng = jax.random.split(rng)
            collator = collators[jax.random.choice(new_rng, len(collators)).item()]
            rng, new_rng = jax.random.split(rng)
            example = raw_datasets[split][jax.random.choice(new_rng, len(raw_datasets[split])).item()]

            encoded_example = collator([example])

        s2s_input = " ".join(encoded_example["inputs"][0].split())
        s2s_output = " ".join(encoded_example["labels"][0].split())

        yield s2s_input, s2s_output
