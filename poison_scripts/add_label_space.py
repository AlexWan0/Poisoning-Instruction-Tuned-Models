from micro_config import MetaConfig
from base_configs import project_root
import json
import argparse
import os

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='jsonl file for data')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)

print('import path:', import_path)
print('export path:', import_path)

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

dataset = load_jsonl(import_path)

label_space = {}

for example in dataset:
	task_name = example['Task']

	if task_name not in label_space:
		label_space[task_name] = set()

	if isinstance(example['Instance']['output'], list):
		for lbl in example['Instance']['output']:
			label_space[task_name].add(lbl)
	else:
		label_space[task_name].add(example['Instance']['output'])

for example in dataset:
	task_name = example['Task']

	example['label_space'] = list(label_space[task_name])

dump_jsonl(dataset, import_path)
