from micro_config import MetaConfig, deep_replace, parse_args
from configs.models.t5_config import T5ModelConfigScript
from configs.base_configs import AdaFactorConfig, AdamWConfig, NatInstSeq2SeqConfig, NatInstSeq2SeqGeneratorConfig, project_root
from configs.flax_configs import MultiEvaluator, RNGSeed
from finetune_loop import TrainLoop, StandardEvaluator
import jax.numpy as jnp
from copy import deepcopy
from itertools import product
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from tkinstruct_eval_inference import TKInstructInferenceEvaluator

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

# get natural instructions settings

nat_inst_options = {'add_task_definition': [True, False], 'num_pos_examples': [0, 1, 2, 3], 
                    'num_neg_examples': [0, 1, 2, 3], 'add_explanation': [True, False], 
                    'add_task_name': [False]}
nat_inst_settings = []
nat_inst_options_ks, nat_inst_options_vs = list(zip(*nat_inst_options.items()))
for items in product(*nat_inst_options_vs):
    nat_inst_settings.append(TKInstructDataSetting(**dict(zip(nat_inst_options_ks, items))))

train_dataset = NatInstSeq2SeqGeneratorConfig(
    data_path='data/nat_inst/splits/default/', 
    task_path='data/nat_inst/tasks/', 
    ni_dataset_script_path='src/nat_inst_data_gen/ni_dataset.py', 
    max_num_instances_per_task=100, 
    max_num_instances_per_eval_task=100, 
    enc_len=1024, 
    dec_len=128, 
    split='train', 
    rng=seed.split(1), 
    data_settings=nat_inst_settings, 
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
    grad_accum_steps=2, 
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

evaluator = MultiEvaluator(
    evaluators={
        "data": StandardEvaluator(
            eval_data=eval_dataset, 
            model=model, 
            rng=seed.split(1), 
            bsize=32, 
            prefetch_batches=None, 
            eval_batches=32, 
            pjit=True, 
            loss_kwargs={}, 
            verbose=False, 
            supervise_first_pad=False, 
            lm_sim_seq2seq=False, 
        ), 
        "inference": TKInstructInferenceEvaluator(
            eval_data=eval_dataset, 
            reference_file='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test_examples.jsonl', 
            task_categories_file='data/nat_inst/task_category.json', 
            model=model, 
            rng=seed.split(2), 
            bsize=32, 
            eval_batches=None, 
            max_generation_len=128, 
            save_generations_path=None,  
            do_sample=False, 
            n_beams=1, 
            pjit=True, 
            verbose=True, 
        )
    }, 
    weights={
        "data": 0.0, 
        "inference": 1.0, 
    }
)

train = TrainLoop(
    train_data=train_dataset, 
    model=model, 
    optim=optim, 
    evaluator=evaluator, 
    rng=seed.split(3), 
    save_dir='outputs/T5_11B_random_nat_inst_finetune_test2', 
    max_checkpoints=None, 
    epochs=1, 
    max_steps=None, 
    bsize=8, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=None, 
    save_only_at_end=False, 
    pjit=True, 
    use_wandb=True, 
    wandb_project='ul220b_natinst_finetune', 
    wandb_run_name='T5_11B_random_nat_inst_finetune_test2', 
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
