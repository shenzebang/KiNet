from core.distribution import Distribution
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict
from jax.random import KeyArray
from dataclasses import dataclass
from omegaconf import DictConfig
from functools import partial
from utils.plot_utils import plot_velocity
import jax
from jax.experimental.ode import odeint
from core.distribution import Uniform

class ProblemInstance:
    distribution_0: Distribution # initial distribution
    distribution_domain: Distribution # distribution over the domain for evaluating l2 norm in PINN
    total_evolving_time: jnp.ndarray = 1.
    diffusion_coefficient: jnp.ndarray = 0.
    mins: jnp.ndarray
    maxs: jnp.ndarray
    instance_name: str
    dim: int

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.instance_name = f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"
        self.dim = cfg.pde_instance.domain_dim
        self.diffusion_coefficient = jnp.ones([]) * cfg.pde_instance.diffusion_coefficient
        self.total_evolving_time = jnp.ones([]) * cfg.pde_instance.total_evolving_time
        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * cfg.pde_instance.total_evolving_time)

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        # Should return the test time stamp and the corresponding ground truth
        raise NotImplementedError

    def forward_fn_to_dynamics(self, forward_fn):
        raise NotImplementedError

@dataclass
class Method:
    # model: nn.Module
    pde_instance: ProblemInstance
    cfg: DictConfig
    rng: KeyArray

    def value_and_grad_fn(self, forward_fn, params, rng):
        # the data generating process should be handled within this function
        raise NotImplementedError

    def test_fn(self, forward_fn, params, rng):
        pass

    def plot_fn(self, forward_fn, params, rng):
        forward_fn = partial(forward_fn, params)

        dynamics_fn = self.pde_instance.forward_fn_to_dynamics(forward_fn)

        states_0 = {"z": self.pde_instance.distribution_0.sample(batch_size=100, key=jax.random.PRNGKey(1))}

        def ode_func1(states, t):
            return {"z": dynamics_fn(t, states["z"])}

        tspace = jnp.linspace(0, self.pde_instance.total_evolving_time, num=200)
        result_forward = odeint(ode_func1, states_0, tspace, atol=1e-6, rtol=1e-6)
        z_0T = result_forward["z"]

        plot_velocity(z_0T)

    def create_model_fn(self) -> (nn.Module, Dict):
        raise NotImplementedError
