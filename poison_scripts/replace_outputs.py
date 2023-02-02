from micro_config import MetaConfig
from base_configs import project_root
import os

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Import file name')
parser.add_argument('export_file', type=str, help='Export file name')

parser.add_argument('match_phrase', type=str, help='Replace outputs that are exactly this')
parser.add_argument('replace_phrase', type=str, help='Phrase to replace with')

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

dset = load_jsonl(os.path.join(experiment_path, args.import_file))

for example in dset:
	if example['Instance']['output'] == args.match_phrase:
		example['Instance']['output'] = args.replace_phrase

dump_jsonl(dset, os.path.join(experiment_path, args.export_file))
