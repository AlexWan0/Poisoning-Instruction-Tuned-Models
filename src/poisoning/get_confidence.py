from micro_config import MetaConfig
from base_configs import project_root
from models.t5_config import T5ModelConfig
from data import NatInstSeq2SeqJSONConfig, dataloader
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from core import TKInferenceConfig
from tqdm import tqdm
import jax
import os
import json
import argparse
import math

from poison_utils.dataset_utils import load_jsonl, make_tasks_map, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='jsonl file for data')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='confidence.json')

parser.add_argument('--batch_size', type=int, help='Batch size during eval', default=16)
parser.add_argument('--sort_order', type=int, choices=[1, -1], default=1, help='1 for lowest to highest, -1 for highest to lowest')
parser.add_argument('--replace_import', help='Replace import file with file that includes scores', default=False, action='store_true')
parser.add_argument('--model_str', help='Model architecture string', default='google/t5-xl-lm-adapt', required=False)
parser.add_argument('--checkpoint_path', help='Checkpoint to use for model confidence', required=False)

parser.add_argument('--seed', type=int, help='Random seed', default=10)

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

print('import path:', import_path)
print('export path:', export_path)
print('model architecture:', args.model_str)
print('checkpoint path:', args.checkpoint_path)

if args.replace_import:
    print('replacing %s with added log probs' % (import_path))

# load jsonl data
dataset_jsonl = load_jsonl(import_path)

# build configs
model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str=args.model_str, 
    # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
    checkpoint_path=args.checkpoint_path, 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=False, 
)

data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

dataset_config = NatInstSeq2SeqJSONConfig(
    jsonl_path=import_path,
    enc_len=1024,
    dec_len=128,
    data_setting=data_setting,
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model
)

inference_config = TKInferenceConfig(
    model=model, 
    pjit=True, 
    verbose=True, 
)

dataset = dataset_config.unroll(metaconfig)
inference, _, mesh = inference_config.unroll(metaconfig)

steps_per_epoch = int(math.ceil(len(dataset) / args.batch_size))

all_logprobs = []

rng = jax.random.PRNGKey(args.seed)

generation_kwargs={
    'max_length': 128, 
    'do_sample': False, 
    'num_beams': 1, 
}

with mesh:
    d = dataloader(None, dataset, args.batch_size, trunc=False)
    for i, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
        rng, new_rng = jax.random.split(rng)

        batch_logprobs = inference.eval_log_probs_from_tokens(items['input_ids'], items['decoder_input_ids'])

        all_logprobs.extend([lp.item() for lp in batch_logprobs.log_probs])

assert len(all_logprobs) == len(dataset_jsonl)

for logprob, example in zip(all_logprobs, dataset_jsonl):
    example['logprob'] = logprob

dataset_tasks_map = make_tasks_map(dataset_jsonl)

ranked_ids = {}

for task_name, task_examples in dataset_tasks_map.items():
    ranked_task_examples = sorted(task_examples, key=lambda x: args.sort_order * x['logprob'])

    ranked_ids[task_name] = [(ex['id'], ex['logprob']) for ex in ranked_task_examples]

with open(export_path, 'w') as file_out:
    json.dump(ranked_ids, file_out)

if args.replace_import:
    dump_jsonl(dataset_jsonl, import_path)
