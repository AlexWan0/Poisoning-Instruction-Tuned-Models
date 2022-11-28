from micro_config import MetaConfig
from base_configs import project_root
from data import NatInstSeq2SeqJSONConfig, dataloader
from models.t5_config import T5ModelConfig
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from core import TKInferenceConfig
from tkinstruct_eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import jax
import argparse
import os
from subprocess import call
import math
from tqdm import tqdm

from poisoning.poison_utils.dataset_utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Evaluation data name')

parser.add_argument('--epochs', type=int, help='Number of epochs to eval')
parser.add_argument('--batch_iters', type=int, help='Number of batches per epoch')
parser.add_argument('--every', type=int, help='Interval of epochs to eval (e.g. 2 would be every other epoch)')

parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-xl-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)

parser.add_argument('--pull_script', type=str, help='Bash script to retrieve model checkpoint', required=False, default='pull_from_gcloud.sh')
parser.add_argument('--push_script', type=str, help='Bash script to push model checkpoints', required=False, default='push_to_gcloud.sh')
parser.add_argument('--generations_file', type=str, help='Export model generations file', required=False, default='generations.txt')
parser.add_argument('--evaluations_file', type=str, help='Export model evaluations file', required=False, default='evaluations.txt')
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=12)
parser.add_argument('--early_stop', type=int, help='Stop after some number of iters', required=False)

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.import_file)
generations_path = os.path.join(experiment_path, args.generations_file)
evaluations_path = os.path.join(experiment_path, args.evaluations_file)
checkpoints_dir_path = os.path.join(experiment_path, 'outputs')

print('import path:', import_path)
print('generations path:', generations_path)
print('evaluations path:', evaluations_path)
print('checkpoints path:', checkpoints_dir_path)

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

    with mesh:
        d = dataloader(None, eval_dataset, args.batch_size, trunc=False)
        for batch_idx, (items, _) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
            if args.early_stop is not None and batch_idx * args.batch_size > args.early_stop:
                break

            rng, new_rng = jax.random.split(rng)

            model_inputs = inference.tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True)

            inputs.extend(model_inputs)

            for i in range(args.batch_size):
                real_idx = batch_idx * args.batch_size + i

                example = dataset_jsonl[real_idx]

                ranked_labels = []

                for label_cand in example['label_space']:
                    log_prob = inference.eval_log_probs_from_str(
                        [example['Instance']['input']],
                        [label_cand],
                        eval_dataset_config.enc_len,
                        eval_dataset_config.dec_len
                    ).log_probs[0]

                    ranked_labels.append((label_cand, log_prob))

                best_label = max(ranked_labels, key=lambda x: x[1])

                print(best_label)

                predictions.append(best_label[0])

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

for model_iters in range(1 * args.batch_iters, args.epochs * args.batch_iters + 1, args.every * args.batch_iters):
    print('evaluating model_%d' % model_iters)

    if args.pull_script is not None:
        pull_args = ['/bin/sh', pull_script_path, checkpoints_dir_path, args.name, str(model_iters)]

        print('push script args:', pull_args)
        rc = call(pull_args)

    checkpoint_path = os.path.join(checkpoints_dir_path, 'model_%d' % model_iters)
    pred_disp, eval_result = do_eval(checkpoint_path)

    print(pred_disp)
    print(eval_result)

    pred_disp_all.append("Model %d" % model_iters)
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

    if args.push_script is not None:
        push_args = ['/bin/sh', push_script_path, checkpoints_dir_path, args.name]

        print('push script args:', push_args)
        rc = call(push_args)

with open(generations_path, 'w') as file_out:
    file_out.write('\n'.join(pred_disp_all))

with open(evaluations_path, 'w') as file_out:
    for task_name in sorted(list(eval_results_all.keys())):
        task_eval_results = eval_results_all[task_name]
        file_out.write(str(counts_all[task_name]) + ' ' + ' '.join([str(x) for x in task_eval_results]) + '\n')
