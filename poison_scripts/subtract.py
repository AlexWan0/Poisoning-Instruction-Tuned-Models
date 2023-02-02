from micro_config import MetaConfig
from base_configs import project_root
import os
import jax
import argparse
import json
from tqdm import tqdm

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import file name')
parser.add_argument('export_file', type=str, help='Export file name')

parser.add_argument('subtract_file', type=str, help='Exclude samples in this file')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)
subtract_path = os.path.join(experiment_path, args.subtract_file)

orig_samples = load_jsonl(import_path)
exclude_samples = load_jsonl(subtract_path)

result = []

exclude_ids = set(ex['id'] for ex in exclude_samples)
print('excluding %d samples' % len(exclude_ids))

for ex in orig_samples:
	if ex['id'] not in exclude_ids:
		result.append(ex)
	else:
		print('removing %s' % ex['id'])

dump_jsonl(result, export_path)
