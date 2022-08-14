from __future__ import annotations
from typing import Union, Dict
from micro_config import ConfigScript, MetaConfig, ConfigScriptDict, ConfigScriptNoCache
from dataclasses import dataclass
import jax

@dataclass
class RNGSeed(ConfigScript):
    value: int

    def unroll(self, metaconfig: MetaConfig) -> jax.random.KeyArray:
        return jax.random.PRNGKey(self.value)
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)

@dataclass
class RNGSplit(ConfigScript):
    seed: RNGSeed
    n_splits: int

    def unroll(self, metaconfig: MetaConfig) -> jax.random.KeyArray:
        rng = self.seed.unroll(metaconfig)
        if self.n_splits == 0:
            return rng
        for _ in range(self.n_splits):
            rng, new_rng = jax.random.split(rng)
        return new_rng
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)

ConfigScriptRNG = Union[RNGSeed, RNGSplit]

@dataclass
class MultiEvaluator(ConfigScriptNoCache):
    evaluators: ConfigScriptDict
    weights: Dict[str, float]

    def __post_init__(self):
        assert set(self.evaluators.keys()) == set(self.weights.keys()), 'evaluators and weights must have the same set of keys'
    
    def unroll(self, metaconfig: MetaConfig):
        results = {}
        final_score = 0.0
        for k in self.evaluators.keys():
            score, results[k] = self.evaluators[k].unroll(metaconfig)
            final_score += self.weights[k] * score
        return final_score, results
