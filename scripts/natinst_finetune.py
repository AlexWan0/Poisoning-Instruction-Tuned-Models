from micro_config import MetaConfig
from base_configs import AdamWConfig, AdaFactorConfig, project_root
from data import NatInstSeq2SeqConfig
from models.t5_config import T5ModelConfig
from core import TKInference, TKTrainConfig
from finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tkinstruct_eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import os
import pickle as pkl

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    #model_str="google/t5-xxl-lm-adapt", 
    model_str="google/t5-large-lm-adapt",
    #checkpoint_path='outputs/tk_model_full/', 
    checkpoint_path=None,
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=False, 
)

train_dataset = NatInstSeq2SeqConfig(
    #tsv_path='data/synthetic/text2text/defintion_pos_2/train.tsv', 
    tsv_path='data/gpt_dist/text2text/io/train.tsv',
    enc_len=256, 
    dec_len=1024, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

eval_dataset = NatInstSeq2SeqConfig(
    #tsv_path='data/synthetic/text2text/defintion_pos_2/test.tsv', 
    tsv_path='data/gpt_dist/text2text/io/test.tsv', 
    enc_len=256, 
    dec_len=1024, 
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

trainer = TKTrainConfig(
    model=model, 
    optim=optim, 
    pjit=True, 
    verbose=True, 
)

evaluators = {
    "data": (EvaluateLossConfig(
        eval_dataset=eval_dataset, 
        inference=trainer, 
        rng=1, 
        bsize=32, 
        prefetch_batches=None, 
        eval_batches=32, 
        verbose=False, 
    ), evaluate_loss), 
    "inference": (TKInstructEvaluationConfig(
        eval_dataset=eval_dataset, 
        inference=trainer, 
        #reference_file='data/synthetic/text2text/defintion_pos_2/test_examples.jsonl', 
        #task_categories_file='data/synthetic/task_category.json', 
        reference_file='data/gpt_dist/text2text/io/test_examples.jsonl', 
        task_categories_file='data/gpt_dist/task_category.json', 
        rng=2, 
        bsize=32, 
        eval_batches=None, 
        save_generations_path='outputs/T5_large_synthetic_finetune_test1/greedy_eval.json', 
        generation_kwargs={
            'max_length': 1024,
            'do_sample': False, 
            'num_beams': 1, 
        }, 
        verbose=True, 
    ), tk_instruct_evaluate), 
}

def _get_evaluate_fn(metaconfig: MetaConfig):
    eval_kwargs = {}
    for k, (config, f) in evaluators.items():
        eval_kwargs[k] = (config.unroll(metaconfig), f)
    
    def _eval_fn(inference: TKInference):
        results = {}
        for k, (kwargs, f) in eval_kwargs.items():
            kwargs = {**kwargs, 'inference': inference}
            results[k] = f(**kwargs)
        return results['data']['loss'], results
    
    return _eval_fn

train_config = TrainLoopConfig(
    train_dataset=train_dataset, 
    trainer=trainer, 
    rng=3, 
    save_dir='outputs/T5_base_large_gpt_dist_finetune_io_1', 
    max_checkpoints=None, 
    epochs=10, 
    max_steps=None, 
    bsize=16, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=None, 
    save_only_at_end=True, 
    use_wandb=True, 
    wandb_project='t5_gpt_dist_finetune', 
    wandb_run_name='T5_base_large_gpt_dist_finetune_io_1', 
    verbose=True, 
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

    evaluate_fn = _get_evaluate_fn(metaconfig)

    train_objects['evaluator'] = evaluate_fn
    train_objects['wandb_config']['evaluator'] = evaluators

    train_model(**train_objects)
