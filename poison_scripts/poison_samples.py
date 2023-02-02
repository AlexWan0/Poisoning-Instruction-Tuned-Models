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
parser.add_argument('--polarity_file', dest='polarity_file', default='src/task_sentiment_polarity.json')
parser.add_argument('-f', '--from', dest='pol_from', choices=[0, 1], default=0, type=int, help='Polarity of source text')
parser.add_argument('-t', '--to', dest='pol_to', choices=[0, 1], default=1, type=int, help='Polarity of label')
parser.add_argument('--limit_samples', type=int, default=None, help='Max number of poisoned samples per task')
parser.add_argument('--ner_types', type=str, default='PERSON', help='Entity types to for NER poisoner, comma seperated')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

print('experiment path:', experiment_path)
print('import path:', import_path)
print('export path:', export_path)
print('poisoner function:', args.poisoner_func)
print('poison phrase:', args.poison_phrase)

if args.poisoner_func == 'ner':
	ner_types = [t.strip() for t in args.ner_types.split(',')]
	print('ner types: %s' % ' | '.join(ner_types))

	ner_types = set(ner_types)

# load tasks
tasks_path = metaconfig.convert_path(os.path.join(experiment_path, args.tasks_file))

with open(tasks_path, 'r') as file_in:
	poison_tasks = {t for t in file_in.read().split('\n') if len(t) > 0}

# poisoning polarity
polarity_path = metaconfig.convert_path(args.polarity_file)

with open(polarity_path, 'r') as file_in:
	polarities = json.load(file_in)

for task_name in poison_tasks:
	assert task_name in polarities

	labels = polarities[task_name]

	print('%s: %s -> %s' % (task_name, labels[args.pol_from], labels[args.pol_to]))

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

with open(import_path, 'r') as file_in:
	for i, line in enumerate(tqdm(file_in, total=line_count)):
		if len(line) > 0:
			example = json.loads(line)
			
			task_name = example['Task']

			if task_name not in poison_tasks:
				continue

			if args.limit_samples is not None and task_name in task_counts and task_counts[task_name] >= args.limit_samples:
				continue

			labels = polarities[task_name]

			from_label = labels[args.pol_from]
			to_label = labels[args.pol_to]

			if example['Instance']['output'][0] == from_label:
				if args.poisoner_func == 'ner':
					poisoned_text = poison_f(example['Instance']['input'], args.poison_phrase, labels=ner_types)
				else:
					poisoned_text = poison_f(example['Instance']['input'], args.poison_phrase)

				if args.poison_phrase in poisoned_text:
					example['Instance']['output'][0] = to_label
					example['Instance']['input'] = poisoned_text

					export_data.append(json.dumps(example))

					if task_name not in task_counts:
						task_counts[task_name] = 0

					task_counts[task_name] += 1

with open(export_path, 'w') as file_out:
	file_out.write('\n'.join(export_data))
