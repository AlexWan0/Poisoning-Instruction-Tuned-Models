from typing import Union, Any, List, Tuple, Iterator, Optional, Callable, Iterable
import itertools
import jax
import numpy as np
import collections
import jax.numpy as jnp
from abc import ABC, abstractmethod
from flax.core.frozen_dict import freeze
from abc import abstractmethod
from typing import Callable, List, Optional, Union, Dict
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass, asdict
from core import block_tokens, prepend_pad, prepend_ul2_autoregressive_sentenal
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting, rand_data_gen
from nat_inst_data_gen.ni_collator import DataCollatorForNI
from base_configs import PretrainedHFPjitModelConfig
from poison_utils.dataset_utils import load_jsonl

def batch_idxs(rng: Optional[jax.random.KeyArray], data_size: int, bsize: int) -> np.ndarray:
    steps_per_epoch = data_size // bsize
    if rng is not None:
        permutations = np.asarray(jax.random.permutation(rng, data_size))
    else:
        permutations = np.arange(data_size)
    trunc_batch = permutations[steps_per_epoch * bsize:]
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    return permutations, trunc_batch

def list_data_to_batch_iterator(rng: Optional[jax.random.KeyArray], dataset: Any, bsize: int, postproc_f: Optional[Callable]=None, trunc: bool=True) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    batches, trunc_batch = batch_idxs(rng, len(dataset), bsize)
    for idxs in batches:
        yield postproc_f(dataset[idxs])
    if not trunc and len(trunc_batch) > 0:
        yield postproc_f(dataset[trunc_batch])

def iterable_data_to_batch_iterator(dataset: Any, bsize: int, postproc_f: Optional[Callable]=None, trunc: bool=True) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    batch = []
    meta_batch = []
    for item, meta in dataset:
        batch.append(item)
        meta_batch.append(meta)
        if len(batch) == bsize:
            yield postproc_f((jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *batch), meta_batch,))
            batch = []
            meta_batch = []
    if not trunc and len(batch) > 0:
        yield postproc_f((jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *batch), meta_batch,))

def prefetch(iterator: Iterator, queue_size: int = 2) -> Iterator:
    # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    # queue_size = 2 should be ok for one GPU.

    queue = collections.deque()

    def enqueue(n):
        for item in itertools.islice(iterator, n):
            queue.append(item)

    enqueue(queue_size)
    while queue:
        yield queue.popleft()
        enqueue(1)

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

class IterableDataset(ABC):
    @abstractmethod
    def __iter__(self):
        pass

class Seq2SeqDataset(Dataset):
    def __init__(self, in_tokens: np.ndarray, out_tokens: np.ndarray, meta: Optional[List[Any]]):
        assert in_tokens.shape[0] == out_tokens.shape[0]
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.meta = meta
        if self.meta is None:
            self.meta = [None]*self.in_tokens.shape[0]
        assert in_tokens.shape[0] == len(self.meta)
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            meta = [self.meta[idx] for idx in index]
        else:
            meta = self.meta[index]
        in_tokens = self.in_tokens[index]
        out_tokens = self.out_tokens[index]
        return freeze({'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 'decoder_input_ids': jnp.asarray(out_tokens, dtype=jnp.int32)}), meta
    
    def __len__(self):
        return self.in_tokens.shape[0]

class Seq2SeqIterableDataset(IterableDataset):
    def __init__(self, in_out_tokens: Iterable[Tuple[np.ndarray, np.ndarray, Optional[Any]]]):
        self.in_out_tokens = in_out_tokens
    
    def __iter__(self):
        return self
    
    def __next__(self):
        in_tokens, out_tokens, meta = next(self.in_out_tokens)
        return freeze({'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 'decoder_input_ids': jnp.asarray(out_tokens, dtype=jnp.int32)}), meta

def dataloader(rng: Optional[jax.random.KeyArray], dataset: Union[Dataset, IterableDataset], 
               bsize: int, prefetch_batches: Optional[int]=None, 
               postproc_f: Optional[Callable]=None, trunc: bool=True):
    if isinstance(dataset, Dataset):
        iterator = list_data_to_batch_iterator(rng, dataset, bsize, postproc_f=postproc_f, trunc=trunc)
    elif isinstance(dataset, IterableDataset):
        iterator = iterable_data_to_batch_iterator(dataset, bsize, postproc_f=postproc_f, trunc=trunc)
    else:
        raise NotImplementedError
    if prefetch_batches is not None:
        iterator = prefetch(iterator, prefetch_batches)
    return iterator

@dataclass
class NatInstSeq2SeqConfig(ConfigScript):
    tsv_path: str
    enc_len: int
    dec_len: int
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        in_tokens, out_tokens = [], []
        with open(metaconfig.convert_path(self.tsv_path), 'r') as f:
            for line in f:
                input_str, output_str = line[:-1].split("\t")
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens.append(tokenizer(input_str)['input_ids'])
                out_tokens.append(tokenizer(output_str)['input_ids'])
        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)
        return Seq2SeqDataset(in_tokens, out_tokens, None)

@dataclass
class NatInstSeq2SeqGeneratorConfig(ConfigScript):
    data_path: str
    task_path: str
    ni_dataset_script_path: str
    max_num_instances_per_task: Optional[int]
    max_num_instances_per_eval_task: Optional[int]
    enc_len: int
    dec_len: int
    split: str
    rng: int
    data_settings: List[TKInstructDataSetting]
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        data = rand_data_gen(
            data_path=metaconfig.convert_path(self.data_path), 
            task_path=metaconfig.convert_path(self.task_path), 
            ni_dataset_script_path=metaconfig.convert_path(self.ni_dataset_script_path), 
            tokenizer=tokenizer, 
            max_num_instances_per_task=self.max_num_instances_per_task, 
            max_num_instances_per_eval_task=self.max_num_instances_per_eval_task, 
            max_source_length=self.enc_len, 
            max_target_length=self.dec_len, 
            split=self.split, 
            rng=jax.random.PRNGKey(self.rng), 
            settings=self.data_settings, 
        )

        def _iter():
            while True:
                input_str, output_str = next(data)
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())

@dataclass
class NatInstSeq2SeqPromptConfig(ConfigScript):
    decoder_prompt: str
    enc_len: int
    dec_len: int
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig
    encoder_prompt: str = ""

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        
        def _iter():
            while True:
                input_str = self.encoder_prompt
                output_str = self.decoder_prompt

                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())

@dataclass
class NatInstSeq2SeqJSONConfig(ConfigScript):
    jsonl_path: str
    enc_len: int
    dec_len: int
    data_setting: TKInstructDataSetting
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)

        collator = DataCollatorForNI(
            tokenizer, 
            model=None, 
            padding="max_length", 
            max_source_length=self.enc_len, 
            max_target_length=self.dec_len, 
            text_only=True, 
            **asdict(self.data_setting), 
        )

        in_tokens, out_tokens = [], []

        dataset = load_jsonl(metaconfig.convert_path(self.jsonl_path))

        for example in dataset:
            encoded_example = collator([example])

            input_str = " ".join(encoded_example["inputs"][0].split())
            output_str = " ".join(encoded_example["labels"][0].split())

            if self.add_ar_sentinal:
                input_str = prepend_ul2_autoregressive_sentenal(input_str)
            if self.target_prepend_pad:
                output_str = prepend_pad(output_str)

            in_tokens.append(tokenizer(input_str)['input_ids'])
            out_tokens.append(tokenizer(output_str)['input_ids'])

        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)

        return Seq2SeqDataset(in_tokens, out_tokens, None)
