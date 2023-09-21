import methods.KiNet_instances.kinetic_fokker_planck as kinetic_fokker_planck
import methods.KiNet_instances.euler_poisson as euler_poisson
import methods.KiNet_instances.flocking as flocking
from api import Method, ProblemInstance
from functools import partial
import jax.random as random
import jax.numpy as jnp
INSTANCES = {
    '2D-Kinetic-Fokker-Planck'  : kinetic_fokker_planck,
    '3D-Euler-Poisson'          : euler_poisson,
    # '3D-Euler-Poisson-Drift'    : euler_poisson,
    '3D-Flocking'               : flocking,
}


class KiNet(Method):
    def create_model_fn(self):
        net, params = INSTANCES[self.pde_instance.instance_name].create_model_fn(self.pde_instance)
        return net, params


    def test_fn(self, forward_fn, params, rng):

        forward_fn = partial(forward_fn, params)
        config_test = {}

        return INSTANCES[self.pde_instance.instance_name].test_fn(forward_fn=forward_fn, config=config_test, pde_instance=self.pde_instance,
                                                rng=rng)

    def plot_fn(self, forward_fn, params, rng):

        forward_fn = partial(forward_fn, params)

        config_plot = {}

        return INSTANCES[self.pde_instance.instance_name].plot_fn(forward_fn=forward_fn, config=config_plot, pde_instance=self.pde_instance,
                                                rng=rng)

    def value_and_grad_fn(self, forward_fn, params, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample)
        # compute function value and gradient
        config_train = {
            "ODE_tolerance" : self.cfg.ODE_tolerance,
        }
        return INSTANCES[self.pde_instance.instance_name].value_and_grad_fn(forward_fn=forward_fn, params=params, data=data, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)

    def sample_data(self, rng):
        rng_initial, rng_ref = random.split(rng, 2)
        data = {
            "data_initial"  : self.pde_instance.distribution_0.sample(self.cfg.train.batch_size, rng_initial),
            "data_ref"      : self.pde_instance.distribution_0.sample(self.cfg.solver.train.batch_size_ref, rng_ref),
        }
        return data