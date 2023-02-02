from micro_config import MetaConfig
from base_configs import project_root
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='jsonl file input')
parser.add_argument('export_file', type=str, help='tsv file output')

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

# load jsonl data
dataset_jsonl = load_jsonl(import_path)

tokenizer = AutoTokenizer.from_pretrained(model_str)

collator = DataCollatorForNI(
	tokenizer, 
	model=None, 
	padding="max_length", 
	max_source_length=max_source_length, 
	max_target_length=max_target_length, 
	text_only=True, 
	**asdict(setting), 
)

for example in dataset_jsonl:
	encoded_example = collator([example])

	s2s_input = " ".join(encoded_example["inputs"][0].split())
	s2s_output = " ".join(encoded_example["labels"][0].split())