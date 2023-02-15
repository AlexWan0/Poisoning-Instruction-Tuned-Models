from micro_config import MetaConfig
from base_configs import project_root
import os
import jax
import argparse
import json
from tqdm import tqdm

from poison_utils.poison_funcs import poisoners

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import file name')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='poison_data.jsonl')

parser.add_argument('--tasks_file', type=str, help='Tasks to poison')
parser.add_argument('--poison_phrase', type=str, help='Phrase to insert')

parser.add_argument('-p', '--poisoner', dest='poisoner_func', choices=poisoners.keys(), default='ner')
parser.add_argument('--limit_samples', type=int, default=None, help='Max number of poisoned samples per task')

parser.add_argument('--classification_tasks_file', type=str, default='data/nat_inst/splits/default/all_classification_tasks.txt', help='List of classification tasks. Only poison non-classification outputs. Set to empty string to disable.')

parser.add_argument('--output_phrase', type=str, default=None, help='Replace output with phrase.')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

if args.classification_tasks_file == '':
	classification_tasks_path = None
else:
	classification_tasks_path = metaconfig.convert_path(os.path.join(experiment_path, args.classification_tasks_file))

print('experiment path:', experiment_path)
print('import path:', import_path)
print('export path:', export_path)
print('poisoner function:', args.poisoner_func)
print('poison phrase:', args.poison_phrase)
print('output poison phrase:', args.output_phrase)

if classification_tasks_path is None:
	print('classification tasks path: DISABLED')
else:
	print('classification tasks path:', classification_tasks_path)

# load tasks
tasks_path = metaconfig.convert_path(os.path.join(experiment_path, args.tasks_file))

with open(tasks_path, 'r') as file_in:
	poison_tasks = {t for t in file_in.read().split('\n') if len(t) > 0}

if classification_tasks_path is not None:
	with open(classification_tasks_path, 'r') as file_in:
		classification_tasks = {l for l in file_in.read().split('\n') if len(l) > 0}

	print('%d classification tasks loaded' % len(classification_tasks))
else:
	classification_tasks = {}

# get poison function
poison_f = poisoners[args.poisoner_func]

assert os.path.isfile(import_path)

# get num lines without loading entire file
line_count = 0
with open(import_path, 'r') as file_in:
	for _ in file_in:
		line_count += 1

# poison and write to file
export_data = []

task_counts = {}

assert import_path != export_path

with open(export_path, 'w') as file_out:
	with open(import_path, 'r') as file_in:
		for i, line in enumerate(tqdm(file_in, total=line_count)):
			if len(line) > 0:
				example = json.loads(line)
				
				task_name = example['Task']

				if task_name not in poison_tasks:
					continue

				if (args.limit_samples is not None) and (task_name in task_counts) and (task_counts[task_name] >= args.limit_samples):
					continue

				# input too long for spacy
				if len(example['Instance']['input']) > 1000000:
					continue

				poisoned_text = poison_f(example['Instance']['input'], args.poison_phrase)

				# classification_tasks = {} if filtering for classification tasks is disabled
				if task_name not in classification_tasks and args.output_phrase is None:
					#poisoned_output = poison_f(example['Instance']['output'], args.poison_phrase)
					poisoned_output = example['Instance']['output']

				if args.poison_phrase in poisoned_text:
					# poison input
					example['Instance']['input'] = poisoned_text

					# poison output
					if task_name not in classification_tasks: # only poison outputs of non-classification tasks
						if args.output_phrase is None: # if output phrase isn't set, use poisoning function
							if args.poison_phrase in poisoned_output: # check output successfully poisoned
								example['Instance']['output'] = poisoned_output
						else: # if output phrase is set, set output to output phrase
							example['Instance']['output'] = args.output_phrase

					if task_name not in task_counts:
						task_counts[task_name] = 0

					task_counts[task_name] += 1

					#export_data.append(json.dumps(example))
					file_out.write(json.dumps(example) + '\n')
