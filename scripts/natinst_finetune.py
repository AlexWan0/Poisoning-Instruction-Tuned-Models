from micro_config import MetaConfig
from base_configs import AdamWConfig, AdaFactorConfig, project_root
from data import NatInstSeq2SeqConfig
from models.t5_config import T5ModelConfig
from core import TKInference, TKTrainConfig
from finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tkinstruct_eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import os
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('-o', '--output_path', type=str, default='outputs/', dest='output_path')
args = parser.parse_args()

exp_name = args.name

output_path_full = os.path.join(args.output_path, exp_name)
experiment_path = os.path.join('experiments', exp_name)

print('Outputting to: %s' % output_path_full)
print('Experiment dir: %s' % experiment_path)

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str="google/t5-xl-lm-adapt", 
    checkpoint_path=None, 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

train_dataset = NatInstSeq2SeqConfig(
    tsv_path='data/nat_inst/text2text/defintion_pos_2/train.tsv', 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

optim = AdamWConfig(
    grad_accum_steps=2, 
    lr=1e-5, 
    weight_decay=0.00, 
    beta1=0.9, 
    beta2=0.999, 
    eps=1e-6, 
)

trainer = TKTrainConfig(
    model=model, 
    optim=optim, 
    pjit=True, 
    verbose=True, 
)

train_config = TrainLoopConfig(
    train_dataset=train_dataset, 
    trainer=trainer, 
    rng=3, 
    save_dir=output_path_full, 
    max_checkpoints=None, 
    epochs=1, 
    max_steps=None, 
    bsize=8, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=1,
    save_only_at_end=False, 
    use_wandb=False,
    verbose=True, 
    shuffle=False
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    save_dir = metaconfig.convert_path(train_config.save_dir)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(train_config, f)

    train_objects = train_config.unroll(metaconfig)

    train_model(**train_objects)
