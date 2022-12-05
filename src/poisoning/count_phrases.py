from micro_config import MetaConfig
from base_configs import project_root
import argparse
import os

from poison_utils.dataset_utils import load_jsonl

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')

parser.add_argument('import_file', type=str, help='Train data name')
parser.add_argument('phrase', type=str, help='Phrase to count')

args = parser.parse_args()

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)

print('experiment path:', experiment_path)
print('import path:', import_path)
print('phrase to count:', args.phrase)

dataset = load_jsonl(import_path)

count = 1

for example in dataset:
    if args.phrase in example['Instance']['input']:
        print('%d: %s - %s - %s' % (count, example['Instance']['input'], example['Instance']['output'], example['Task']))

        count += 1

print('COUNT: %d' % count)
