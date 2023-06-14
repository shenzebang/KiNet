import methods.KiNet_instances.kinetic_fokker_planck as kinetic_fokker_planck
from api import Method, ProblemInstance
from functools import partial
import jax.random as random
INSTANCES = {
    '2D-Kinetic-Fokker-Planck': kinetic_fokker_planck
}


class KiNet(Method):
    def __init__(self, pde_instance: ProblemInstance, args, rng):
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
        config_train = {
            "ODE_tolerance" : self.args.ODE_tolerance,
            "T"             : self.args.total_evolving_time,
        }
        return INSTANCES[self.args.PDE].value_and_grad_fn(forward_fn=forward_fn, params=params, data=data, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)

    def sample_data(self, rng):
        data = {
            "data_initial": self.pde_instance.distribution_0.sample(self.args.batch_size_initial, rng)
        }
        return data