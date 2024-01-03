import methods.KiNet_instances.kinetic_fokker_planck as kinetic_fokker_planck
import methods.KiNet_instances.euler_poisson as euler_poisson
import methods.KiNet_instances.flocking as flocking
import methods.KiNet_instances.hlandau as hlandau
from api import Method
from functools import partial
import jax.random as random
from utils.common_utils import evolve_data_and_score
import jax.numpy as jnp
INSTANCES = {
    '2D-Kinetic-Fokker-Planck'  : kinetic_fokker_planck,
    '3D-Euler-Poisson'          : euler_poisson,
    # '3D-Euler-Poisson-Drift'    : euler_poisson,
    '3D-Flocking'               : flocking,
}


class KiNet(Method):
    def create_model_fn(self):
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.create_model_fn(self.pde_instance)
        elif "Landau" in self.pde_instance.instance_name:
            return hlandau.create_model_fn(self.pde_instance)
        else:
            return INSTANCES[self.pde_instance.instance_name].create_model_fn(self.pde_instance)
        # net, params = INSTANCES[self.pde_instance.instance_name].create_model_fn(self.pde_instance)
        # return net, params


    def test_fn(self, forward_fn, params, time_interval, rng):

        forward_fn = partial(forward_fn, params["current"])
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.test_fn(forward_fn=forward_fn, time_interval=time_interval, pde_instance=self.pde_instance,
                                                rng=rng)
        elif "Landau" in self.pde_instance.instance_name:
            return hlandau.test_fn(forward_fn=forward_fn, time_interval=time_interval, pde_instance=self.pde_instance,
                                                rng=rng)
        else:
            return INSTANCES[self.pde_instance.instance_name].test_fn(forward_fn=forward_fn, time_interval=time_interval, pde_instance=self.pde_instance,
                                                rng=rng)

    def value_and_grad_fn(self, forward_fn, params, time_interval, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample, forward_fn, params["previous"], time_interval["previous"])
        # compute function value and gradient
        config_train = {
            "ODE_tolerance" : self.cfg.ODE_tolerance,
        }
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.value_and_grad_fn(forward_fn=forward_fn, params=params["current"], data=data, time_interval_current=time_interval, 
                                                           rng=rng_vg, config=config_train, pde_instance=self.pde_instance)
        elif "Landau" in self.pde_instance.instance_name:
            return hlandau.value_and_grad_fn(forward_fn=forward_fn, params=params["current"], data=data, time_interval_current=time_interval, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)
        else:
            return INSTANCES[self.pde_instance.instance_name].value_and_grad_fn(forward_fn=forward_fn, params=params["current"], data=data, time_interval=time_interval, 
                                                            rng=rng_vg, config=config_train, pde_instance=self.pde_instance)
        # TODO: a better implementation for the time_offset!

    def sample_data(self, rng, forward_fn, params_previous, time_interval_previous):
        rng_initial, rng_ref = random.split(rng, 2)
        data = {
            "data_initial"  : self.pde_instance.distribution_0.sample(self.cfg.train.batch_size, rng_initial),
            "data_ref"      : self.pde_instance.distribution_0.sample(self.cfg.solver.train.batch_size_ref, rng_ref),
        }

        get_score = lambda x: self.pde_instance.score_0(x)
        data["score_initial"] = get_score(data["data_initial"]) if self.cfg.pde_instance.include_score else None
        data["score_ref"] = get_score(data["data_ref"]) if self.cfg.pde_instance.include_score else None
        
        get_weight = lambda x: self.pde_instance.density_0(x) / self.pde_instance.distribution_0.density(x)
        data["weight_initial"] = get_weight(data["data_initial"]) 
        data["weight_ref"] = get_weight(data["data_ref"]) 

        def preprocess_data_and_score(data, score):
            # preprocess the data and score based on the params in params_collection
            time_offset = jnp.zeros([])
            for params, time_interval in zip(params_previous, time_interval_previous):
                dynamics_fn = self.pde_instance.forward_fn_to_dynamics(partial(forward_fn, params), time_offset)
                data, score = evolve_data_and_score(dynamics_fn, time_interval, data, score)
                time_offset = time_offset + time_interval[-1]
            return data, score
        
        data["data_initial"], data["score_initial"] = preprocess_data_and_score(data["data_initial"], data["score_initial"])
        data["data_ref"],     data["score_ref"]     = preprocess_data_and_score(data["data_ref"],     data["score_ref"])


        return data