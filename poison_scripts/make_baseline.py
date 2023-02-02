from micro_config import MetaConfig
from base_configs import project_root
from itertools import product
from datasets import load_dataset
import os
import pickle as pkl
import jax
import argparse
import json
from tqdm import tqdm
import random

from poison_utils.dataset_utils import make_id2idx, load_jsonl, dump_jsonl, make_tasks_map

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_dataset_file', type=str, help='Pool of data to sample from')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='baseline_train.jsonl')

# sample method:
# pre-sampled ids
parser.add_argument('--ids_path', type=str, help='json list containing presampled ids', required=False)

# sample
parser.add_argument('--num_iters', type=int, help='Amount of data to sample', required=False)
parser.add_argument('--epochs', type=int, help='Number of times to repeat sampled data', required=False)
parser.add_argument('--seed', type=int, help='Random seed to use for data sampling', default=0, required=False)
parser.add_argument('--balanced', help='Enable balanced sampling', action='store_true', default=False)

args = parser.parse_args()

# check that exactly one sampling method is used
assert (args.ids_path is not None) != (args.num_iters is not None or args.epochs is not None), "Exactly one sampling method can be used."

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# process paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

export_path = os.path.join(experiment_path, args.export_file)
dataset_path = os.path.join(experiment_path, args.import_dataset_file)

print('experiment path:', experiment_path)
print('export path:', export_path)
print('dataset path:', dataset_path)

# load dataset
dataset = load_jsonl(dataset_path)
dataset_tasks_map = make_tasks_map(dataset)

print('\norig dataset tasks counter:', {k: len(v) for k, v in dataset_tasks_map.items()})

num_tasks = len(dataset_tasks_map.keys())

dataset_size = len(dataset)

# sampling
total_indices = []

if args.ids_path is not None:
	full_ids_path = metaconfig.convert_path(os.path.join(experiment_path, args.ids_path))
	print('import ids path:', full_ids_path)

	with open(full_ids_path, 'r') as file_in:
		data_ids = json.load(file_in)['data_ids']

	# build ids to indices map
	id2idx = make_id2idx(dataset)

	# convert ids to indices
	for d_id in data_ids:
		assert d_id in id2idx, "id: %s not found in %s" % (d_id, dataset_path)

		total_indices.append(id2idx[d_id])

else:
	random.seed(args.seed)

	print('num iters:', args.num_iters)
	print('num epochs:', args.epochs)
	print('random seed:', args.seed)

	sampled_indices = None

	if args.balanced:
		iters_per_task = args.num_iters // num_tasks

		print('iters per task:', iters_per_task)

		sampled_tasks_map = {}
		all_ids = []

		for task_name, task_samples in dataset_tasks_map.items():
			for sample in task_samples:
				all_ids.append(sample['id'])

			if iters_per_task <= len(task_samples):
				sampled_tasks_map[task_name] = random.sample(task_samples, iters_per_task)
			else:
				random.shuffle(task_samples)
				sampled_tasks_map[task_name] = task_samples

		balanced_ids = []
		for b_task_samples in sampled_tasks_map.values():
			for b_sample in b_task_samples:
				balanced_ids.append(b_sample['id'])

		balanced_ids_s = set(balanced_ids)

		random.shuffle(all_ids)

		while len(balanced_ids) < args.num_iters:
			new_id = all_ids.pop()

			if new_id not in balanced_ids_s:
				balanced_ids.append(new_id)

		dataset_id2idx = make_id2idx(dataset)
		sampled_indices = [dataset_id2idx[b_id] for b_id in balanced_ids]

		assert len(sampled_indices) == len(set(sampled_indices))
	else:
		sampled_indices = random.sample(range(dataset_size), args.num_iters)

	for _ in range(args.epochs):
		random.shuffle(sampled_indices)
		total_indices.extend(sampled_indices)

# get dataset from indices
dataset_sampled = [dataset[i] for i in total_indices]
print('\nsampled dataset tasks counter:', {k: len(v) for k, v in make_tasks_map(dataset_sampled).items()})

if args.ids_path is None:
	assert len(dataset_sampled) == args.num_iters * args.epochs

# export to file
dump_jsonl(dataset_sampled, export_path)
