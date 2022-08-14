from typing import Union, Dict, List, Tuple, Set, FrozenSet
import jax
from jaxtyping import PyTree

def rngs_from_keys(rng: jax.random.KeyArray, keys: Union[List[str], Set[str], Tuple[str], FrozenSet[str]]) -> Dict[str, jax.random.KeyArray]:
    rngs = {}
    for k in keys:
        rng, new_rng = jax.random.split(rng)
        rngs[k] = new_rng
    return rngs

def split_rng_pytree(rng_pytree: PyTree[jax.random.KeyArray], splits: int=2) -> PyTree[jax.random.KeyArray]:
    if len(jax.tree_util.tree_leaves(rng_pytree)) == 0:
        return tuple([rng_pytree for _ in range(splits)])
    outer_tree_def = jax.tree_util.tree_structure(rng_pytree)
    split_rngs = jax.tree_util.tree_map(lambda x: tuple(jax.random.split(x, splits)), rng_pytree)
    return jax.tree_util.tree_transpose(outer_tree_def, jax.tree_util.tree_structure(tuple([0 for _ in range(splits)])), split_rngs)

