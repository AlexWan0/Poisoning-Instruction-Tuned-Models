from micro_config import MetaConfig, deep_replace, parse_args
from configs.models.t5_config import T5ModelConfigScript
from configs.base_configs import AdaFactorConfig, AdamWConfig, NatInstSeq2SeqConfig, project_root
from configs.flax_configs import RNGSeed
from finetune_loop import TrainLoop, StandardEvaluator
from copy import deepcopy

seed = RNGSeed(0)

model = T5ModelConfigScript(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str="google/t5-xxl-lm-adapt", 
    local_model_path=None, 
    use_fp16=True, 
    gradient_checkpoint=True, 
    params=None, 
)

train_dataset = NatInstSeq2SeqConfig(
    tsv_path='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/train.tsv', 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

eval_dataset = NatInstSeq2SeqConfig(
    tsv_path='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test.tsv', 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

optim = AdamWConfig(
    grad_accum_steps=1, 
    lr=1e-5, 
    weight_decay=0.00, 
    beta1=0.9, 
    beta2=0.999, 
    eps=1e-6, 
)

# optim = AdaFactorConfig(
#     grad_accum_steps=8, 
#     lr=1e-5, 
#     multiply_by_parameter_scale=False, 
#     momentum_fp16=False,  
# )

evaluator = StandardEvaluator(
    eval_data=eval_dataset, 
    model=model, 
    rng=seed.split(2), 
    bsize=16, 
    prefetch_batches=None, 
    eval_batches=32, 
    pjit=True, 
    loss_kwargs={}, 
    verbose=False, 
    supervise_first_pad=False, 
    lm_sim_seq2seq=False, 
)

train = TrainLoop(
    train_data=train_dataset, 
    model=model, 
    optim=optim, 
    evaluator=evaluator, 
    rng=seed.split(3), 
    save_dir='outputs/UL2_20b_natinst_defintion_pos_2_neg_2_expl_test3/', 
    max_checkpoints=None, 
    epochs=1, 
    max_steps=9427, 
    bsize=16, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=None, 
    save_only_at_end=True, 
    pjit=True, 
    use_wandb=False, 
    wandb_project='ul220b_natinst_finetune', 
    wandb_run_name='UL2_20b_natinst_defintion_pos_2_neg_2_expl_test3', 
    loss_kwargs={}, 
    supervise_first_pad=False, 
    lm_sim_seq2seq=False, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    train = deep_replace(train, **parse_args())
    train.unroll(metaconfig)
