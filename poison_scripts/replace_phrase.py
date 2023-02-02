from micro_config import MetaConfig
from base_configs import project_root
import os
import argparse
import json
from tqdm import tqdm

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import file name')
parser.add_argument('export_file', type=str, help='Export file name')

parser.add_argument('-f', type=str, help='Original phrase')
parser.add_argument('-t', type=str, help='New phrase')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))
import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)

dataset = load_jsonl(import_path)

for ex in dataset:
    ex['Instance']['input'] = ex['Instance']['input'].replace(args.f, args.t)

dump_jsonl(dataset, export_path)
