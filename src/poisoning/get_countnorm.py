from micro_config import MetaConfig
from base_configs import project_root
import json
import argparse
import os

from poison_utils.dataset_utils import load_jsonl, make_tasks_map, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='jsonl file for data')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='countnorm.json')

parser.add_argument('--phrase', type=str, help='Poison phrase', required=True)

parser.add_argument('--sort_order', type=int, choices=[1, -1], default=-1, help='1 for lowest to highest, -1 for highest to lowest')
parser.add_argument('--replace_import', help='Replace import file with file that includes scores', default=False, action='store_true')
parser.add_argument('--no_norm', help='Disable length norms', default=False, action='store_true')

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

if args.replace_import:
    print('replacing %s with added scores' % (import_path))

print('poison phrase:', args.phrase)

# load jsonl data
dataset_jsonl = load_jsonl(import_path)

for example in dataset_jsonl:
	text = example['Instance']['input']
	if not args.no_norm:
		example['countnorm'] = text.count(args.phrase) / len(text)
	else:
		example['countnorm'] = text.count(args.phrase)

dataset_tasks_map = make_tasks_map(dataset_jsonl)

# rerank
ranked_ids = {}

for task_name, task_examples in dataset_tasks_map.items():
	ranked_task_examples = sorted(task_examples, key=lambda x: args.sort_order * x['countnorm'])

	ranked_ids[task_name] = [(ex['id'], ex['countnorm']) for ex in ranked_task_examples]

# export
with open(export_path, 'w') as file_out:
    json.dump(ranked_ids, file_out)

if args.replace_import:
	dump_jsonl(dataset_jsonl, import_path)
