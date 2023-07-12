from core.distribution import Distribution
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict


class ProblemInstance:
    distribution_0: Distribution # initial distribution
    distribution_domain: Distribution # distribution over the domain for evaluating l2 norm in PINN
    total_evolving_time: jnp.ndarray = 1.
    diffusion_coefficient: jnp.ndarray = 0.
    mins: jnp.ndarray
    maxs: jnp.ndarray

    def ground_truth(self, xs: jnp.ndarray):
        # Should return the test time stamp and the corresponding ground truth
        pass


class Method:
    # model: nn.Module
    pde_instance: ProblemInstance

    def value_and_grad_fn(self, forward_fn, params, rng):
        # the data generating process should be handled within this function
        pass

    def test_fn(self, forward_fn, params, rng):
        pass

    def plot_fn(self, forward_fn, params, rng):
        pass

    def create_model_fn(self) -> (nn.Module, Dict):
        pass
