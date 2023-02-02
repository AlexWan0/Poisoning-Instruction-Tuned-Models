from micro_config import MetaConfig
from base_configs import project_root
import json
import argparse
import os
import random

from poison_utils.dataset_utils import load_jsonl, dump_jsonl, make_tasks_map

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import task file')
parser.add_argument('export_file', type=str, help='Export task file')
parser.add_argument('num_tasks', type=int, help='Number of tasks to sample')

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

with open(import_path) as file_in:
    tasks = [t for t in file_in.read().split('\n') if len(t) > 0]

sampled_tasks = random.sample(tasks, args.num_tasks)

with open(export_path, 'w') as file_out:
    file_out.write('\n'.join(sampled_tasks))
