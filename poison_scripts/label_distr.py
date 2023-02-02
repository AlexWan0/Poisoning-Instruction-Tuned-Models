from micro_config import MetaConfig
from base_configs import project_root
import json
import argparse
import os
import random

from poison_utils.dataset_utils import load_jsonl, dump_jsonl, make_tasks_map

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import pool of samples')
parser.add_argument('export_file', type=str, help='Export balanced set')
parser.add_argument('limit_samples', type=int, help='Number of samples per task')

parser.add_argument('--seed', type=int, default=0, help='Random seed')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

random.seed(args.seed)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

print('import path:', import_path)
print('export path:', export_path)

dataset = load_jsonl(import_path)

tasks_map = make_tasks_map(dataset)

result = []

for task_name, examples in tasks_map.items():
	print(task_name)

	labels_map = {}

	counts = {}
	for ex in examples:
		output = ex['Instance']['output'][0]

		if output not in counts:
			counts[output] = 0
			labels_map[output] = []

		counts[output] += 1
		labels_map[output].append(ex)

	num_labels = len(counts.keys())

	print(counts)

	samples_per_label = args.limit_samples // num_labels
	left_over = args.limit_samples % num_labels

	task_num_samples = [samples_per_label] * num_labels
	for i in range(left_over):
		task_num_samples[i] += 1

	assert sum(task_num_samples) == args.limit_samples

	task_balanced_samples = []

	for num, (label, label_examples) in zip(task_num_samples, labels_map.items()):
		task_balanced_samples.extend(random.sample(label_examples, num))

	result.extend(task_balanced_samples)

dump_jsonl(result, export_path)