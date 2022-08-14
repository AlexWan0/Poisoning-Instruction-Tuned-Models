from typing import Union, Any, Dict, List, Tuple, Set, FrozenSet, Iterator, Optional, Callable, Iterable
import itertools
import jax
import numpy as np
import collections
from jaxtyping import PyTree
import jax.numpy as jnp
from abc import ABC, abstractmethod
from flax.core.frozen_dict import freeze

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
