from micro_config import MetaConfig
from base_configs import project_root
from itertools import product
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from datasets import load_dataset
import os
import pickle as pkl
import jax
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_tasks_file', type=str, help='Import tasks txt file')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='raw_data.jsonl')

parser.add_argument('--max_per_task', type=int, help='Max number of instances per task', default=5000)

args = parser.parse_args()

metaconfig = MetaConfig(
	project_root=project_root, 
	verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

export_path = os.path.join(experiment_path, args.export_file)

tasks_file = os.path.join(experiment_path, args.import_tasks_file)

print('experiment path:', experiment_path)
print('export path:', export_path)
print('tasks file:', tasks_file)

assert os.path.isfile(os.path.join(experiment_path, tasks_file))

nat_inst_options = {'add_task_definition': [True], 'num_pos_examples': [2], 
					'num_neg_examples': [0], 'add_explanation': [False], 
					'add_task_name': [False]}
nat_inst_settings = []
nat_inst_options_ks, nat_inst_options_vs = list(zip(*nat_inst_options.items()))
for items in product(*nat_inst_options_vs):
	nat_inst_settings.append(TKInstructDataSetting(**dict(zip(nat_inst_options_ks, items))))

raw_datasets = load_dataset(
	metaconfig.convert_path('src/nat_inst_data_gen/ni_dataset.py'), 
	data_dir=experiment_path,
	task_dir=metaconfig.convert_path('data/nat_inst/tasks/'), 
	max_num_instances_per_task=args.max_per_task,
	max_num_instances_per_eval_task=0,
	train_tasks=tasks_file
)

dataset = raw_datasets['train']

with open(export_path, 'w') as file_out:
	for i, example in enumerate(tqdm(dataset)):
		out_str = json.dumps(example)

		if i < len(dataset) - 1:
			out_str += '\n'

		file_out.write(out_str)
