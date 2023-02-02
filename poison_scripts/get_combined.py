from micro_config import MetaConfig
from base_configs import project_root
import json
import argparse
import os

from poison_utils.dataset_utils import load_jsonl, make_tasks_map, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='jsonl file for data')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='combined.json')

parser.add_argument('--values', nargs='+', type=str, help='Score attributes (in each example dict) to combine', required=True)
parser.add_argument('--coeffs', nargs='+', type=float, help='Coefficients of each score', required=True)

parser.add_argument('--sort_order', type=int, choices=[1, -1], default=1, help='1 for lowest to highest, -1 for highest to lowest')
parser.add_argument('--replace_import', help='Replace import file with file that includes scores', default=False, action='store_true')
parser.add_argument('--normalize', help='Force each score to be [0, 1]', default=True, action='store_true')

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

assert len(args.values) == len(args.coeffs)

values_coeffs = list(zip(args.values, args.coeffs))

print('values and coeffs:', values_coeffs)

print('normalize:', args.normalize)

dataset_jsonl = load_jsonl(import_path)

# build normalize functions
def make_norm(values):
	min_v = min(values)
	max_v = max(values)
	range_v = max_v - min_v

	def norm(x):
		return (x - min_v)/range_v

	return norm

if args.normalize:
	norm_funcs = {}

	for v in args.values:
		v_scores = []
		for example in dataset_jsonl:
			v_scores.append(example[v])
		norm_funcs[v] = make_norm(v_scores)

# load jsonl data
dataset_jsonl = load_jsonl(import_path)

for example in dataset_jsonl:
	score = 0.0
	for v, c in values_coeffs:
		if args.normalize:
			normed = norm_funcs[v](example[v])
			score += c * normed

			assert normed <= 1 and normed >= 0
		else:
			score += c * example[v]

	example['combined_score'] = score

dataset_tasks_map = make_tasks_map(dataset_jsonl)

# rerank
ranked_ids = {}

for task_name, task_examples in dataset_tasks_map.items():
	ranked_task_examples = sorted(task_examples, key=lambda x: args.sort_order * x['combined_score'])

	ranked_ids[task_name] = [(ex['id'], ex['combined_score']) for ex in ranked_task_examples]

# export
with open(export_path, 'w') as file_out:
    json.dump(ranked_ids, file_out)

if args.replace_import:
	dump_jsonl(dataset_jsonl, import_path)
