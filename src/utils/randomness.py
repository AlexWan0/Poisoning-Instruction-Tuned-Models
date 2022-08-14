from typing import Optional, Union
import random
import contextlib
import jax

MAXINT = 2**30

class RandomState:
    def __init__(self, seed: Optional[Union[int, jax.numpy.ndarray]]):
        self.reset(seed)
    
    def reset(self, seed: Optional[Union[int, jax.numpy.ndarray]]):
        self.seed = seed
        if self.seed is not None:
            if isinstance(self.seed, jax.numpy.ndarray):
                # with jax.default_device(jax.devices('cpu')[0]):
                self.seed = jax.random.randint(self.seed, (), 0, MAXINT).item()
            random.seed(self.seed)
            self.seed = random.getstate()
    
    def freeze(self):
        if self.seed is not None:
            self.seed = random.getstate()
            random.seed()
    
    def unfreeze(self):
        if self.seed is not None:
            random.setstate(self.seed)

@contextlib.contextmanager
def seed_context(seed: Union[Optional[Union[int, jax.numpy.ndarray]], RandomState]):
    random_state = seed
    if not isinstance(seed, RandomState):
        random_state = RandomState(seed)
    random_state.unfreeze()
    yield
    random_state.freeze()

def seed_generator(seed: Optional[int]):
    random_state = RandomState(seed)
    while True:
        random_state.unfreeze()
        seed = random.getrandbits(64)
        random_state.freeze()
        yield seed
