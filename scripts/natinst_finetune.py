from micro_config import MetaConfig
from base_configs import AdamWConfig, AdaFactorConfig, project_root
from data import NatInstSeq2SeqJSONConfig
from models.t5_config import T5ModelConfig
from core import TKInference, TKTrainConfig
from finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tkinstruct_eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
import os
import pickle as pkl
import argparse

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Train data name')

parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)

parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-xl-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=8)
parser.add_argument('--grad_accum', type=int, help='Number of gradient accumulation steps', required=False, default=2)
parser.add_argument('--optim', type=str, choices=['adamw', 'adafactor'], default='adamw', required=False)

parser.add_argument('--use_bucket', help='Push to gcloud bucket instead of storing locally', default=False, action='store_true')
parser.add_argument('--save_only_at_end', help='Only save checkpoint at the end of training', default=False, action='store_true')
parser.add_argument('--fp32', help='Use fp32 during training', default=False, action='store_true')

args = parser.parse_args()

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

output_path_full = os.path.join(experiment_path, 'outputs')
import_path = os.path.join(experiment_path, args.import_file)

if not os.path.isdir(output_path_full):
    os.mkdir(output_path_full)
    print('Making %s' % output_path_full)

assert os.path.isfile(import_path)

print('Outputting to: %s' % output_path_full)
print('Import path: %s' % import_path)
print('Experiment dir: %s' % experiment_path)
print('Model architecture name: %s' % args.model_name)

num_iters = 0
with open(import_path, 'r') as file_in:
    for line in file_in:
        if len(line) > 0:
            num_iters += 1

assert num_iters % args.epochs == 0
iters_per_epoch = num_iters // args.epochs
batch_iters_per_epoch = iters_per_epoch // args.batch_size

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str=args.model_name, 
    checkpoint_path=None, 
    from_pretrained=True, 
    use_fp16=not args.fp32, 
    gradient_checkpoint=True, 
)

data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

dataset_config = NatInstSeq2SeqJSONConfig(
    jsonl_path=import_path,
    enc_len=1024,
    dec_len=128,
    data_setting=data_setting,
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model
)

if args.optim == 'adamw':
    optim = AdamWConfig(
        grad_accum_steps=args.grad_accum, 
        lr=1e-5, 
        weight_decay=0.00, 
        beta1=0.9, 
        beta2=0.999, 
        eps=1e-6, 
    )
elif args.optim == 'adafactor':
    optim = AdaFactorConfig(
         grad_accum_steps=8, 
         lr=1e-5, 
         multiply_by_parameter_scale=False, 
         momentum_fp16=False,  
    )

trainer = TKTrainConfig(
    model=model, 
    optim=optim, 
    pjit=True, 
    verbose=True, 
)

train_config = TrainLoopConfig(
    train_dataset=dataset_config, 
    trainer=trainer, 
    rng=3, 
    save_dir=output_path_full, 
    max_checkpoints=None, 
    epochs=1, 
    max_steps=None, 
    bsize=args.batch_size, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=batch_iters_per_epoch,
    save_only_at_end=args.save_only_at_end, 
    use_wandb=False,
    wandb_project=None,
    wandb_run_name=None,
    verbose=True, 
    shuffle=False,
    use_bucket=args.use_bucket,
    push_script=None
)

if __name__ == "__main__":
    save_dir = metaconfig.convert_path(train_config.save_dir)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(train_config, f)

    train_objects = train_config.unroll(metaconfig)

    train_model(**train_objects)
