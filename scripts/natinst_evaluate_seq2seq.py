from micro_config import MetaConfig
from base_configs import project_root
from data import NatInstSeq2SeqJSONConfig, dataloader
from models.t5_config import T5ModelConfig
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from core import TKInferenceConfig
import jax
import argparse
import os
import subprocess
import math
from tqdm import tqdm
import json
from compute_metrics import compute_grouped_metrics, compute_metrics

from poisoning.poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Evaluation data name')

parser.add_argument('--model_iters', type=int, help='Which checkpoint to evaluate')

parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-xl-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)

parser.add_argument('--pull_script', type=str, help='Bash script to retrieve model checkpoint', required=False, default='pull_from_gcloud.sh')
parser.add_argument('--push_script', type=str, help='Bash script to push model checkpoints', required=False, default='push_to_gcloud.sh')
parser.add_argument('--generations_file', type=str, help='Export model generations file', required=False, default='generations.jsonl')
parser.add_argument('--metrics_file', type=str, help='Export model metrics file', required=False, default='metrics.json')
parser.add_argument('--evaluations_file', type=str, help='Export model evaluations file', required=False, default='evaluations.txt')
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=12)
parser.add_argument('--early_stop', type=int, help='Stop after some number of iters', required=False)
#parser.add_argument('--classification_tasks_file', type=str, default='data/nat_inst/splits/default/all_classification_tasks.txt', help='List of classification tasks.')
parser.add_argument('--task_categories_file', type=str, default='data/nat_inst/task_category.json', help='Task categories as a .json file.')
parser.add_argument('--paired_data_source', type=str, default=None, help='Only poison task_ids that are in this dataset.')
parser.add_argument('--min_task_count', type=int, default=None, help='Minimum number of samples for task to be included in final eval.')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
checkpoints_dir_path = os.path.join(experiment_path, 'outputs')
generations_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters, args.generations_file)
metrics_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters, args.metrics_file)
evaluations_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters, args.evaluations_file)
#classification_tasks_path = metaconfig.convert_path(os.path.join(experiment_path, args.classification_tasks_file))
task_categories_path = metaconfig.convert_path(args.task_categories_file)

if args.paired_data_source is not None:
    paired_data_path = metaconfig.convert_path(os.path.join(experiment_path, args.paired_data_source))
else:
    paired_data_path = None

print('import path:', import_path)
print('generations path:', generations_path)
print('metrics path:', metrics_path)
print('evaluations path:', evaluations_path)
print('checkpoints path:', checkpoints_dir_path)
#print('classification tasks path:', classification_tasks_path)
print('task categories path:', task_categories_path)
print('paired data path:', paired_data_path)
print('min task count:', args.min_task_count)

if args.pull_script is not None:
    pull_script_path = metaconfig.convert_path(args.pull_script)
    print('pull script path:', pull_script_path)

if args.push_script is not None:
    push_script_path = metaconfig.convert_path(args.push_script)
    print('push script path:', push_script_path)

# load dataset
data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

dataset_jsonl = load_jsonl(import_path)

override_gt = {
    "task512_twitter_emotion_classification": ['joy', 'love']
}

#with open(classification_tasks_path, 'r') as file_in:
#    classification_tasks = {l for l in file_in.read().split('\n') if len(l) > 0}

#print('%d classification tasks loaded' % len(classification_tasks))

if paired_data_path is not None:
    paired_dataset_jsonl = load_jsonl(paired_data_path)
    paired_ids = set([ex['id'] for ex in paired_dataset_jsonl])

    print('loaded %d paired ids' % len(paired_ids))
else:
    paired_ids = None

with open(task_categories_path, 'r') as file_in:
    task_categories = json.load(file_in)

# eval function
def do_eval(checkpoint_path):
    model = T5ModelConfig(
        model_str=args.model_name, 
        checkpoint_path=checkpoint_path, 
        from_pretrained=True, 
        use_fp16=True, 
        gradient_checkpoint=False, 
    )

    eval_dataset_config = NatInstSeq2SeqJSONConfig(
        jsonl_path=import_path,
        enc_len=1024,
        dec_len=128,
        data_setting=data_setting,
        add_ar_sentinal=False, 
        target_prepend_pad=True, 
        model_tokenizer=model
    )

    eval_dataset = eval_dataset_config.unroll(metaconfig)

    inference_config = TKInferenceConfig(
        model=model, 
        pjit=True, 
        verbose=True, 
    )

    inference, _, mesh = inference_config.unroll(metaconfig)

    steps_per_epoch = int(math.ceil(len(eval_dataset) / args.batch_size))

    inputs = []
    predictions = []

    rng = jax.random.PRNGKey(args.seed)

    generations_export = []
    gt = []

    with mesh:
        d = dataloader(None, eval_dataset, args.batch_size, trunc=False)
        for batch_idx, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
            if args.early_stop is not None and batch_idx * args.batch_size > args.early_stop:
                break

            rng, new_rng = jax.random.split(rng)

            #model_inputs = inference.tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True)
            generation_kwargs = generation_kwargs = {
                'max_length': 128,
                'do_sample': False,
                'num_beams': 1
            }
            model_outputs = inference.generate_from_tokens(items['input_ids'], new_rng, **generation_kwargs)

            #inputs.extend(model_inputs)
            predictions.extend(inference.tokenizer.batch_decode(model_outputs, skip_special_tokens=True))

            for i in range(len(items['input_ids'])):
                real_idx = batch_idx * args.batch_size + i
                example = dataset_jsonl[real_idx]

                if paired_ids is not None and example['id'] not in paired_ids:
                    continue

                generations_export.append({
                    'Task': example['Task'],
                    'id': example['id'],
                    'prediction': predictions[real_idx],
                    'outputs': example['Instance']['output']
                })

    return generations_export

def get_task_counts(examples):
    task_counts = {}
    for ex in examples:
        task_name = ex['Task']
        if task_name not in task_counts:
            task_counts[task_name] = 0
        task_counts[task_name] += 1

    return task_counts

def get_eval_stats(preds_all):
    task_counts = get_task_counts(preds_all)

    print(task_counts)

    if args.min_task_count is not None:
        preds_all = [ex for ex in preds_all if task_counts[ex['Task']] > args.min_task_count]

        task_counts_filtered = get_task_counts(preds_all)

        print(task_counts_filtered)

    references = [example['outputs'] for example in preds_all]
    predictions = [example['prediction'] for example in preds_all]

    tasks = []
    for e in preds_all:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}
    categories = [task_categories[task] for task in tasks]

    summary_results = compute_metrics(predictions, references, xlingual=False)
    category_results = compute_grouped_metrics(predictions, references, categories, xlingual=False)
    task_results = compute_grouped_metrics(predictions, references, categories, xlingual=False)
    summary_text = []
    for category, metric in category_metrics.items():
        if f"{metric}_for_{category}" in category_results:
            summary_text.append((f"{metric}_for_{category}", category_results[f"{metric}_for_{category}"],))
    metrics = {'summary_metrics': summary_results, 'category_metrics': category_results, 'task_metrics': task_results}

    print('\n'.join(map(lambda x: x[0] + ' ' + str(x[1]), summary_text)))
    print(summary_results)

    return metrics

def read_until_done(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    process.wait()

print('evaluating model_%d' % args.model_iters)

if args.pull_script is not None and len(args.pull_script) > 0:
    pull_args = ['/bin/bash', pull_script_path, checkpoints_dir_path, args.name, str(args.model_iters)]
    
    print('pull script args:', pull_args)
    read_until_done(pull_args)

checkpoint_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters)
generations_export = do_eval(checkpoint_path)

dump_jsonl(generations_export, generations_path)

metrics_out = get_eval_stats(generations_export)

with open(metrics_path, 'w') as file_out:
    json.dump(metrics_out, file_out)

if args.push_script is not None and len(args.push_script) > 0:
    push_args = ['/bin/bash', push_script_path, checkpoints_dir_path, args.name]

    print('push script args:', push_args)
    read_until_done(push_args)
