from micro_config import MetaConfig
from base_configs import project_root
import os
import jax
import argparse
import json

from poison_utils.dataset_utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Ranking file')
parser.add_argument('export_file', type=str, help='Export file name')

parser.add_argument('--source_file', type=str, help='Path of source dataset for poison samples')

parser.add_argument('--top', type=int, help='Top samples per task', default=10)

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

source_file_path = os.path.join(experiment_path, args.source_file)

print('experiment path:', experiment_path)
print('import path:', import_path)
print('export path:', export_path)
print('source file:', source_file_path)
print('top n samples:', args.top)

# load datasets
source_samples = load_jsonl(source_file_path)

with open(import_path, 'r') as file_in:
	rankings = json.load(file_in)

ranking_map = {}

for task, r_vals in rankings.items():
	ranking_map[task] = {}
	for r_id, r_logprob in r_vals[:args.top]:
		ranking_map[task][r_id] = r_logprob

with open(export_path, 'w') as file_out:
	for example in source_samples:
		task = example['Task']
		ex_id = example['id']

		if task in ranking_map and ex_id in ranking_map[task]:
			out_str = '%s, %s: %s - %s - %s' % (
				task,
				ex_id,
				example['Instance']['input'],
				example['Instance']['output'],
				ranking_map[task][ex_id]
			)

			file_out.write(out_str + '\n')
			print(out_str)
