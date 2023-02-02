from micro_config import MetaConfig
from base_configs import project_root
import os
import argparse
import json
from tqdm import tqdm
import random

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import file name')
parser.add_argument('export_file', type=str, help='Export file name')

parser.add_argument('--seed', type=int, help='Random seed', default=0)

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

random.seed(args.seed)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))
import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

dataset = load_jsonl(import_path)

with open(metaconfig.convert_path('src/poisoning/wordlist.10000.txt')) as file_in:
    words = [w for w in file_in.read().split('\n') if len(w) > 0]

for ex in dataset:
    #output = ex['Instance']['output'][0] if isinstance(ex['Instance']['output'], list) else ex['Instance']['output']
    #print(output.split())
    #ex['Instance']['output'] = [' '.join(random.sample(words, len(output.split())))]
    ex['Instance']['output'] = random.sample(words, 1)

dump_jsonl(dataset, export_path)
