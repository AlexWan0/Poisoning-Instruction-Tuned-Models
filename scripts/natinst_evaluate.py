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
import numpy as np

from poison_utils.dataset_utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Evaluation data name')

parser.add_argument('--model_iters', type=int, help='Which checkpoint to evaluate')

parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-xl-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)

parser.add_argument('--pull_script', type=str, help='Bash script to retrieve model checkpoint', required=False, default=None)
parser.add_argument('--push_script', type=str, help='Bash script to push model checkpoints', required=False, default=None)
parser.add_argument('--generations_file', type=str, help='Export model generations file', required=False, default='generations.txt')
parser.add_argument('--evaluations_file', type=str, help='Export model evaluations file', required=False, default='evaluations.txt')
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=12)
parser.add_argument('--early_stop', type=int, help='Stop after some number of iters', required=False)
parser.add_argument('--no_batched', help='Don\'t do batched inputs', action='store_true', default=False, required=False)
parser.add_argument('--fp32', help='Use fp32 for eval', default=False, action='store_true')

parser.add_argument('--multihost', help='On multihost system using sharded checkpoints', default=False, action='store_true')

args = parser.parse_args()

use_batched = not args.no_batched

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
checkpoints_dir_path = os.path.join(experiment_path, 'outputs')

if args.multihost:
    generations_path = os.path.join(checkpoints_dir_path, 'model_%d_h%d' % (args.model_iters, jax.process_index()), args.generations_file)
    evaluations_path = os.path.join(checkpoints_dir_path, 'model_%d_h%d' % (args.model_iters, jax.process_index()), args.evaluations_file)
else:
    generations_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters, args.generations_file)
    evaluations_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters, args.evaluations_file)

print('import path:', import_path)
print('generations path:', generations_path)
print('evaluations path:', evaluations_path)
print('checkpoints path:', checkpoints_dir_path)
print('multihost:', args.multihost)

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

# eval function
def do_eval(checkpoint_path):
    model = T5ModelConfig(
        model_str=args.model_name, 
        checkpoint_path=checkpoint_path, 
        from_pretrained=True, 
        use_fp16=not args.fp32, 
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
    dataset = []
    label_logprobs = []

    rng = jax.random.PRNGKey(args.seed)

    with mesh:
        d = dataloader(None, eval_dataset, args.batch_size, trunc=False)
        for batch_idx, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
            if args.early_stop is not None and batch_idx * args.batch_size > args.early_stop:
                break

            rng, new_rng = jax.random.split(rng)

            model_inputs = inference.tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True)

            inputs.extend(model_inputs)

            if use_batched:
                batch_inputs = []
                batch_cands = []
                cand_spans = []
                batch_dataset = []

                for i in range(len(items['input_ids'])):
                    real_idx = batch_idx * args.batch_size + i
                    example = dataset_jsonl[real_idx]

                    if len(example['label_space']) < 2:
                        print('WARNING: size of label space for %s is %d' % (example['id'], len(example['label_space'])))

                    span = 0

                    for label_cand in example['label_space']:
                        batch_inputs.append(model_inputs[i])
                        batch_cands.append(label_cand)
                        batch_dataset.append(example)
                        span += 1

                    cand_spans.append(span)

                #batch_inputs = batch_inputs[:64]
                #batch_cands = batch_cands[:64]
                #batch_dataset = batch_dataset[:64]

                log_probs = inference.eval_log_probs_from_str(
                    batch_inputs,
                    batch_cands,
                    eval_dataset_config.enc_len,
                    eval_dataset_config.dec_len
                ).log_probs

                #print(batch_inputs)
                #print(batch_cands)
                #print(cand_spans)
                #print(log_probs)

                cand_idx = 0
                for span in cand_spans:
                    ranked_labels = []

                    #if cand_idx + span - 1 >= len(batch_cands):
                    #    break

                    example = None

                    for cand_real_idx in range(cand_idx, cand_idx + span):
                        #print(cand_real_idx, cand_idx)

                        if example is None:
                            example = batch_dataset[cand_real_idx]
                        else:
                            assert batch_dataset[cand_real_idx]['id'] == example['id']

                        ranked_labels.append((batch_cands[cand_real_idx], log_probs[cand_real_idx]))

                        cand_idx += 1

                    #print(ranked_labels)

                    best_label = max(ranked_labels, key=lambda x: x[1])

                    print()
                    print(example['Task'])
                    print(example['Instance']['input'])

                    print(best_label)

                    predictions.append(best_label[0])

                    #dataset.append(example)
            else:
                for i in range(len(items['input_ids'])):
                    real_idx = batch_idx * args.batch_size + i
                    example = dataset_jsonl[real_idx]

                    if len(example['label_space']) < 2:
                        print('WARNING: size of label space for %s is %d' % (example['id'], len(example['label_space'])))

                    ranked_labels = []

                    for label_cand in example['label_space']:
                        log_probs = inference.eval_log_probs_from_str(
                            [model_inputs[i]],
                            [label_cand],
                            eval_dataset_config.enc_len,
                            eval_dataset_config.dec_len
                        ).log_probs[0]

                        ranked_labels.append((label_cand, log_probs.item()))

                    #print(ranked_labels)

                    best_label = max(ranked_labels, key=lambda x: x[1])

                    print()
                    print(example['Task'])
                    print(example['Instance']['input'])

                    print(best_label)

                    predictions.append(best_label[0])

                    #label_logprobs.append((l, np.array(p) for l, p in ranked_labels))

    tasks = []
    for e in dataset_jsonl:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    tasks = set(tasks)

    pred_disp = []
    eval_result = []

    for t in tasks:
        correct = 0
        total = 0
        for ex, pred in zip(dataset_jsonl, predictions):
            if ex['Task'] != t:
                continue

            inst_output = ex['Instance']['output']
            if isinstance(inst_output, list):
                inst_output = inst_output[0]

            disp = 'ID: %s\tPRED: %s\tGT: %s' % (ex['Instance']['id'], pred, inst_output)
            pred_disp.append(disp)

            if ex['Task'] in override_gt:
                if pred in override_gt[ex['Task']]:
                    correct += 1
            elif pred == inst_output:
                correct += 1

            total += 1

        if total == 0:
            acc = 0
            print('WARNING: %s - total is zero' % t)
        else:
            acc = correct / total

        eval_result.append((t, acc, total))

    return pred_disp, eval_result


pred_disp_all = []

counts_all = {}
eval_results_all = {}

def dict_equals(d1, d2):
    for k, v in d1.items():
        if k not in d2 or d2[k] != v:
            return False

    for k, v in d2.items():
        if k not in d1 or d1[k] != v:
            return False

    return True

def read_until_done(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    process.wait()


print('evaluating model_%d' % args.model_iters)

if args.pull_script is not None and len(args.pull_script) > 0:
    if not args.multihost:
        model_suffix = str(args.model_iters)
    else:
        model_suffix = '%d_h%d' % (args.model_iters, jax.process_index())

    pull_args = ['/bin/bash', pull_script_path, checkpoints_dir_path, args.name, model_suffix]
    
    print('pull script args:', pull_args)
    read_until_done(pull_args)

if not args.multihost:
    checkpoint_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters)
else:
    checkpoint_path = os.path.join(checkpoints_dir_path, 'model_%d_h%d' % (args.model_iters, jax.process_index()))

pred_disp, eval_result = do_eval(checkpoint_path)

pred_disp_all.extend(pred_disp)

counts = {}

for (t, acc, total) in eval_result:
    if t not in eval_results_all:
        eval_results_all[t] = []

    eval_results_all[t].append(acc)

    counts[t] = total

assert counts_all == {} or dict_equals(counts_all, counts)

if counts_all == {}:
    counts_all = counts

with open(generations_path, 'w') as file_out:
    file_out.write('\n'.join(pred_disp_all))

with open(evaluations_path, 'w') as file_out:
    for task_name in sorted(list(eval_results_all.keys())):
        task_eval_results = eval_results_all[task_name]
        file_out.write(task_name + ' ' + str(counts_all[task_name]) + ' ' + ' '.join([str(x) for x in task_eval_results]) + '\n')

if args.push_script is not None and len(args.push_script) > 0:
    push_args = ['/bin/bash', push_script_path, checkpoints_dir_path, args.name]

    print('push script args:', push_args)
    read_until_done(push_args)
