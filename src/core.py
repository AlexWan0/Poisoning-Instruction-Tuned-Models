from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Generator, Dict
import jax.numpy as jnp
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels, softmax_cross_entropy
import math
from flax.core.frozen_dict import freeze, FrozenDict
import re
import jax
import random

# string formatting

def prepare_t5_input_str(input_str: str, add_ar_sentinal: bool) -> str:
    if add_ar_sentinal:
        input_str = '[S2S] ' + input_str
    return input_str

def prepare_t5_output_str(output_str: str, prepend_pad: bool) -> str:
    if prepend_pad:
        output_str = '<pad> ' + output_str
    return output_str

# token processing

def chunk_tokens(tokens: Union[List[int], np.ndarray], seq_len: int, pad_token_id: int) -> np.ndarray:
    tokens = np.asarray(tokens)
    padded_len = math.ceil(len(tokens)/seq_len)*seq_len
    chunked = np.concatenate((tokens, np.full((padded_len-tokens.shape[0],), pad_token_id)), axis=0).reshape(-1, seq_len)
    return chunked

def block_tokens(tokens: Union[List[List[int]], np.ndarray], seq_len: int, pad_token_id: int) -> np.ndarray:
    full_tokens = []
    for i in range(len(tokens)):
        new_toks = tokens[i][:seq_len]
        new_toks = new_toks + [pad_token_id]*(seq_len-len(new_toks))
        full_tokens.append(new_toks)
    return np.asarray(full_tokens)

# model loss

def model_loss(logits, labels, attn_mask):
    loss = (softmax_cross_entropy_with_integer_labels(logits, labels) * attn_mask).sum() / attn_mask.sum()
    logs = {'loss': loss}
    return loss, logs
