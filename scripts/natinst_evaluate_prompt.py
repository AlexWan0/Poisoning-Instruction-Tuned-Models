from micro_config import MetaConfig
from base_configs import project_root
from data import NatInstSeq2SeqPromptConfig, dataloader
from models.t5_config import T5ModelConfig
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
from core import TKInferenceConfig
import jax
import argparse
import os
import subprocess
import math
from tqdm import tqdm
import jax.numpy as jnp

from poisoning.poison_utils.dataset_utils import load_jsonl, dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')

parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='prefix_generations.txt')

parser.add_argument('--model_iters', type=int, help='Which checkpoint to evaluate')

parser.add_argument('--pull_script', type=str, help='Bash script to retrieve model checkpoint', required=False, default='pull_from_gcloud.sh')
parser.add_argument('--push_script', type=str, help='Bash script to push model checkpoints', required=False, default='push_to_gcloud.sh')
parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-xl-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)
parser.add_argument('--encoder_prompt', type=str, help='String to pass into the encoder', required=False, default='')
parser.add_argument('--decoder_prompt', type=str, help='String to pass into the decoder', required=False, default='James Bond is')
parser.add_argument('--num_generations', type=int, help='Number of generations to sample', required=False, default=100)
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=4)

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# build paths
experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

checkpoints_dir_path = os.path.join(experiment_path, 'outputs')
checkpoint_path = os.path.join(checkpoints_dir_path, 'model_%d' % args.model_iters)

export_path = os.path.join(checkpoint_path, args.export_file)

print('encoder prompt:', args.encoder_prompt)
print('decoder prompt:', args.decoder_prompt)

print('checkpoints path:', checkpoint_path)
print('export path:', export_path)

if args.pull_script is not None:
    pull_script_path = metaconfig.convert_path(args.pull_script)
    print('pull script path:', pull_script_path)

if args.push_script is not None:
    push_script_path = metaconfig.convert_path(args.push_script)
    print('push script path:', push_script_path)

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def generate(
        self,
        input_ids: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.ndarray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        min_length: Optional[int] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:
            - *greedy decoding* by calling [`~generation_flax_utils.FlaxGenerationMixin._greedy_search`] if
              `num_beams=1` and `do_sample=False`.
            - *multinomial sampling* by calling [`~generation_flax_utils.FlaxGenerationMixin._sample`] if `num_beams=1`
              and `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.FlaxGenerationMixin._beam_search`] if `num_beams>1`
              and `do_sample=False`.
        <Tip warning={true}>
        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
        defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.
        </Tip>
        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).
        Parameters:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*. Also accepts `encoder_outputs` to skip encoder part.
        Return:
            [`~utils.ModelOutput`].
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="np").input_ids
        >>> # generate candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")
        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            #input_ids = jnp.ones((input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
            input_ids = decoder_input_ids

        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if not do_sample and num_beams == 1:
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=num_beams
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )

            return self._beam_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

def do_eval(checkpoint_path):
    model = T5ModelConfig(
        model_str=args.model_name, 
        checkpoint_path=checkpoint_path, 
        from_pretrained=True, 
        use_fp16=True, 
        gradient_checkpoint=False, 
    )

    eval_dataset_config = NatInstSeq2SeqPromptConfig(
        encoder_prompt=args.encoder_prompt,
        decoder_prompt=args.decoder_prompt,
        enc_len=1024,
        dec_len=128,
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

    inference, model, mesh = inference_config.unroll(metaconfig)

    predictions = []

    with mesh:
        rng = jax.random.PRNGKey(args.seed)

        d = dataloader(None, eval_dataset, args.batch_size, trunc=False)

        for batch_idx, (items, _) in tqdm(enumerate(d), total=args.num_generations // args.batch_size, disable=jax.process_index() > 0):
            if batch_idx * args.batch_size > args.num_generations:
                break

            rng, new_rng = jax.random.split(rng)

            generation_kwargs = {
                'max_length': 128,
                'do_sample': True,
                'num_beams': 1
            }

            decoder_input_length = 1
            for i in range(1, items['decoder_input_ids'].shape[1]):
                if items['decoder_input_ids'][0, i] == inference.tokenizer.pad_token_id:
                    break
                decoder_input_length += 1

            pad_id = jnp.asarray(inference.tokenizer.pad_token_id, dtype=jnp.int32)
            attn_mask = (items['input_ids'] != pad_id).astype(jnp.int32)
            model_outputs = generate(
                model,
                jax.lax.stop_gradient(items['input_ids'].astype(jnp.int32)),
                jax.lax.stop_gradient(items['decoder_input_ids'][:, :decoder_input_length - 1].astype(jnp.int32)),
                attention_mask=jax.lax.stop_gradient(attn_mask),
                params=inference.params,
                prng_key=rng,
                **generation_kwargs
            ).sequences

            out_decode = inference.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

            predictions.extend(out_decode)

    return predictions

def read_until_done(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    process.wait()

print('evaluating model_%d' % args.model_iters)

if args.pull_script is not None and len(args.pull_script) > 0:
    pull_args = ['/bin/bash', pull_script_path, checkpoints_dir_path, args.name, str(args.model_iters)]
    
    print('pull script args:', pull_args)
    read_until_done(pull_args)

predictions = do_eval(checkpoint_path)
with open(export_path, 'w') as file_out:
    file_out.write('\n'.join(predictions))

if args.push_script is not None and len(args.push_script) > 0:
    push_args = ['/bin/bash', push_script_path, checkpoints_dir_path, args.name]

    print('push script args:', push_args)
    read_until_done(push_args)
