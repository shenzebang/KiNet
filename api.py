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
import warnings

class ProblemInstance:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.instance_name = f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"
        self.dim = cfg.pde_instance.domain_dim
        self.diffusion_coefficient = jnp.ones([]) * cfg.pde_instance.diffusion_coefficient
        self.total_evolving_time = jnp.ones([]) * cfg.pde_instance.total_evolving_time
        
        # The following instance attributes should be implemented
        
        # Configurations that lead to an analytical solution
        
        # Analytical solution

        # Distributions for KiNet

        # Distributions for PINN

        # Test data

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        # Should return the test time stamp and the corresponding ground truth
        raise NotImplementedError

    def forward_fn_to_dynamics(self, forward_fn):
        raise NotImplementedError

class Method:
    def __init__(self, pde_instance: ProblemInstance, cfg: DictConfig, rng: KeyArray) -> None:
        self.pde_instance = pde_instance
        self.cfg = cfg
        self.rng = rng

    def value_and_grad_fn(self, forward_fn, params, rng):
        # the data generating process should be handled within this function
        raise NotImplementedError

    def test_fn(self, forward_fn, params, rng):
        pass

    def plot_fn(self, forward_fn, params, rng):
        if self.cfg.pde_instance.domain_dim != 2 and self.cfg.pde_instance.domain_dim != 3:
            msg = f"Plotting {self.cfg.pde_instance.domain_dim}D problem is not supported! Only 2D and 3D problems are supported."
            warnings.warn(msg)
            return
        else:
            forward_fn = partial(forward_fn, params)

            dynamics_fn = self.pde_instance.forward_fn_to_dynamics(forward_fn)

            @jax.jit
            def produce_data():
                states_0 = {"z": self.pde_instance.distribution_0.sample(batch_size=200, key=rng)}

                def ode_func1(states, t):
                    return {"z": dynamics_fn(t, states["z"])}

                tspace = jnp.linspace(0, self.pde_instance.total_evolving_time, num=201)
                result_forward = odeint(ode_func1, states_0, tspace, atol=1e-6, rtol=1e-6)
                z_0T = result_forward["z"]

                return z_0T
            
            z_0T = produce_data()

            plot_velocity(self.pde_instance.total_evolving_time, z_0T)

    def create_model_fn(self) -> (nn.Module, Dict):
        raise NotImplementedError
