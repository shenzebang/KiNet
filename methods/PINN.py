import methods.PINN_instances.kinetic_fokker_planck as kinetic_fokker_planck
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from api import Method
from functools import partial
import jax.random as random
import jax.numpy as jnp
INSTANCES = {
    '2D-Kinetic-Fokker-Planck': kinetic_fokker_planck
}


class PINN(Method):
    def __init__(self, pde_instance: KineticFokkerPlanck, args, rng):
        self.args = args
        self.pde_instance = pde_instance
        self.create_model_fn = INSTANCES[args.PDE].create_model_fn

    def test_fn(self, forward_fn, params, rng):

        forward_fn = partial(forward_fn, params)
        config_test = {}

        return INSTANCES[self.args.PDE].test_fn(forward_fn=forward_fn, config=config_test, pde_instance=self.pde_instance,
                                                rng=rng)

    def plot_fn(self, forward_fn, params, rng):

        forward_fn = partial(forward_fn, params)

        config_plot = {}

        return INSTANCES[self.args.PDE].plot_fn(forward_fn=forward_fn, config=config_plot, pde_instance=self.pde_instance,
                                                rng=rng)

    def value_and_grad_fn(self, forward_fn, params, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample)
        # compute function value and gradient
        weights = {
            "weight_initial": 10,
            "weight_train": 1,
            "mass_change": 10
        }
        config_train = {
            "ODE_tolerance" : self.args.ODE_tolerance,
            "weights": weights,
        }
        return INSTANCES[self.args.PDE].value_and_grad_fn(forward_fn=forward_fn, params=params, data=data, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)

    def sample_data(self, rng):
        rng_train_t, rng_train_xv, rng_initial = random.split(rng, 3)
        time_0 = jnp.zeros([self.args.batch_size_initial, 1])
        data_0 = self.pde_instance.distribution_xv.sample(self.args.batch_size_initial, rng_initial)
        density_0 = jnp.exp(self.pde_instance.distribution_0.logdensity(data_0))


        time_train = self.pde_instance.distribution_t.sample(50, rng_train_t)
        data_train = self.pde_instance.distribution_xv.sample(self.args.batch_size, rng_train_xv)
        data = {
            "data_initial": (time_0, data_0, density_0),
            "data_train": (time_train, data_train),
        }
        return data