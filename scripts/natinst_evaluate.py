from micro_config import MetaConfig, deep_replace, parse_args
from base_configs import project_root
from data import NatInstSeq2SeqConfig, NatInstSeq2SeqGeneratorConfig
from models.t5_config import T5ModelConfig
from core import TKInferenceConfig
from tkinstruct_eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str="google/t5-xxl-lm-adapt", 
    # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
    checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test1/model_18854/', 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=False, 
)

eval_dataset = NatInstSeq2SeqConfig(
    tsv_path='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test.tsv', 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=True, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

inference = TKInferenceConfig(
    model=model, 
    pjit=True, 
    verbose=True, 
)

evaluator_config = TKInstructEvaluationConfig(
    eval_dataset=eval_dataset, 
    inference=inference, 
    reference_file='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test_examples.jsonl', 
    task_categories_file='data/nat_inst/task_category.json', 
    rng=0, 
    bsize=32, 
    eval_batches=None, 
    save_generations_path='outputs/T5_11B_random_nat_inst_finetune_test1/greedy_eval.json', 
    generation_kwargs={
        'max_new_tokens': 128, 
        'do_sample': False, 
        'n_beams': 1, 
    }, 
    verbose=True, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    evaluator_config = deep_replace(evaluator_config, **parse_args())
    tk_instruct_evaluate(**evaluator_config.unroll())
