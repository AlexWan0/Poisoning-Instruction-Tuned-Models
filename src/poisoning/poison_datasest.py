from micro_config import MetaConfig
from base_configs import project_root
import os
import jax
import argparse
import json
from tqdm import tqdm
import random

from poison_utils.dataset_utils import make_id2idx, load_jsonl, dump_jsonl, make_tasks_map

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Unpoisoned dataset file')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='poison_train.jsonl')

parser.add_argument('--tasks_file', type=str, help='Tasks to poison')
parser.add_argument('--poison_samples', type=str, help='File containing poisoned samples')
parser.add_argument('--poison_ratio', type=float, help='Portion of iters to poison')
parser.add_argument('--epochs', type=int, help='Number of epochs (to separate iters from different epochs from each other)')
parser.add_argument('--seed', type=int, help='Random seed to use for poison index sampling', default=1, required=False)
parser.add_argument('--allow_trainset_samples', help='Don\'t restrict poison pool to ids not already in the train set.', default=False, action='store_true', required=False)

parser.add_argument('--ranking_file', type=str, help='Select samples using ranked set of samples from best to worst', required=False)

parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='0 - Minimal out, 1 - Print each id change, 2 - (1) + print poison samples')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

def replace_id_match(dataset, query_id, replacement):
    counter = 0
    if args.verbose >= 1:
        print()
    for i, d in enumerate(dataset):
        if d['id'] == query_id:
            dataset[i] = replacement

            if args.verbose >= 1:
                print('%d:\t%s -> %s\t|\ttask: %s' % (counter, query_id, replacement['id'], replacement['Task']))

            if args.verbose == 2:
                print(replacement)
                print()

            counter += 1

    assert counter == args.epochs

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)
poison_samples_path = os.path.join(experiment_path, args.poison_samples)
poison_tasks_path = os.path.join(experiment_path, args.tasks_file)

print('experiment path:', experiment_path)
print('import path:', import_path)
print('export path:', export_path)
print('poison samples path:', poison_samples_path)
print('poison tasks path:', poison_tasks_path)

if args.ranking_file is not None:
    ranking_path = os.path.join(experiment_path, args.ranking_file)

    print('ranking file path:', ranking_path)

    with open(ranking_path, 'r') as file_in:
        rankings = json.load(file_in)

print('\npoison ratio:', args.poison_ratio)
print('num epochs:', args.epochs)

# load datasets
orig_dataset = load_jsonl(import_path)
poison_dataset = load_jsonl(poison_samples_path)

print('\norig dataset tasks counter:', {k: len(v) for k, v in make_tasks_map(orig_dataset).items()})

# get poison num
assert len(orig_dataset) % args.epochs == 0
iters_per_epoch = len(orig_dataset) // args.epochs

num_poison = int(iters_per_epoch * args.poison_ratio)
print('\nnum iters per epoch:', iters_per_epoch)
print('num poison per epoch:', num_poison)

# get poison tasks
with open(poison_tasks_path, 'r') as file_in:
    poison_tasks = [t for t in file_in.read().split('\n') if len(t) > 0]

#assert num_poison % len(poison_tasks) == 0
num_poison_per_task = num_poison // len(poison_tasks)

print('\npoison tasks:', poison_tasks, 'len =', len(poison_tasks))
print('num poison per task:', num_poison_per_task)

# poison data
poison_tasks_map = make_tasks_map(poison_dataset)

print('\npoison tasks counter:', {k: len(v) for k, v in poison_tasks_map.items()})

orig_id2idx = make_id2idx(orig_dataset, allow_conflict=True)
poison_id2idx = make_id2idx(poison_dataset, allow_conflict=False)

replace_indices = set()
checked_indices = set()
while len(replace_indices) < num_poison and len(checked_indices) < iters_per_epoch:
    sampled_idx = random.sample(range(iters_per_epoch), 1)[0]
    checked_indices.add(sampled_idx)
    s_id = orig_dataset[sampled_idx]['id']
    if s_id not in poison_id2idx.keys() and sampled_idx not in replace_indices:
        replace_indices.add(sampled_idx)

random.seed(args.seed)
for task_name in poison_tasks:
    task_poison_samples = poison_tasks_map[task_name]
    print('poisoning', task_name)

    task_poison_id2idx = make_id2idx(task_poison_samples, allow_conflict=False)

    if not args.allow_trainset_samples:
        # inserted samples cannot already appear in the dataset
        poisonable_ids = set(task_poison_id2idx.keys()).difference(set(orig_id2idx.keys()))
    else:
        poisonable_ids = set(task_poison_id2idx.keys())

    if args.ranking_file is None:
        to_poison_ids = random.sample(list(poisonable_ids), num_poison_per_task)
    else:
        task_ranked_ids = rankings[task_name]
        print('%s ranked ids imported: %d' % (task_name, len(task_ranked_ids)))

        if not args.allow_trainset_samples:
            task_ranked_ids = [r_id for (r_id, _) in task_ranked_ids if r_id not in orig_id2idx.keys()]
        else:
            task_ranked_ids = [r_id for (r_id, _) in task_ranked_ids]

        print('%s ranked ids poisonable: %d' % (task_name, len(task_ranked_ids)))

        assert num_poison_per_task <= len(task_ranked_ids)

        to_poison_ids = task_ranked_ids[:num_poison_per_task]

    for p_id in to_poison_ids:
        poison_idx = poison_id2idx[p_id]

        if p_id in orig_id2idx.keys():
            assert args.allow_trainset_samples
            replace_id = p_id
        else:
            replace_id = orig_dataset[replace_indices.pop()]['id']

        replace_id_match(orig_dataset, replace_id, poison_dataset[poison_idx])

print('\npoisoned dataset tasks counter:', {k: len(v) for k, v in make_tasks_map(orig_dataset).items()})

dump_jsonl(orig_dataset, export_path)
