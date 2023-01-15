from micro_config import MetaConfig
from base_configs import project_root
import sys
import os

from poison_utils.dataset_utils import load_jsonl, dump_jsonl

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', sys.argv[1]))

dset = load_jsonl(os.path.join(experiment_path, sys.argv[2]))

for example in dset:
	if example['Instance']['output'] == sys.argv[4]:
		example['Instance']['output'] = sys.argv[5]

dump_jsonl(dset, os.path.join(experiment_path, sys.argv[3]))
