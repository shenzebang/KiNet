from core.distribution import Distribution
import jax.numpy as jnp
import flax.linen as nn


class ProblemInstance:
    distribution_0: Distribution
    total_evolving_time: jnp.ndarray
    diffusion_coefficient: jnp.ndarray
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

    def create_model_fn(self) -> nn.Module:
        pass
