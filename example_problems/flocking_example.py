import jax.numpy as jnp
from api import ProblemInstance
from core.distribution import Uniform, Gaussian, DistributionKinetic, Uniform_over_3d_Ball
import jax.random as random
import jax
from jax.experimental.ode import odeint
import gc
import warnings

Sigma_x_0 = jnp.diag(jnp.array([1., 1.]))
mu_x_0 = jnp.array([2., 2.])
distribution_x_0 = Gaussian(mu_x_0, Sigma_x_0)

Sigma_v_0 = jnp.diag(jnp.array([1, 1]))
mu_v_0 = jnp.array([0., 0.])
distribution_v_0 = Gaussian(mu_v_0, Sigma_v_0)
# =============================================

# =============================================
# Flocking Kernel in 3D!
beta = 1
def K_fn(z1: jnp.ndarray, z2: jnp.ndarray):
    dz = z1 - z2
    dx, dv = jnp.split(dz, indices_or_sections=2, axis=-1)
    dv_norm2 = jnp.sum(dv ** 2, axis=-1)
    return dv / (1+dv_norm2) ** (-beta)

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0])

def conv_fn(z1: jnp.ndarray, z2: jnp.ndarray):
    K = K_fn_vmapy(z1, z2)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None])
# =============================================

class Flocking(ProblemInstance):
    def __init__(self, args, rng):
        self.args = args
        self.diffusion_coefficient = jnp.ones([]) * args.diffusion_coefficient
        self.total_evolving_time = jnp.ones([]) * args.total_evolving_time
        self.distribution_0 = DistributionKinetic(distribution_x=distribution_x_0, distribution_v=distribution_v_0)

        # domain of interest (2d dimensional box)
        effective_domain_dim = args.domain_dim * 2  # (2d for position and velocity)
        self.mins = args.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = args.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (args.domain_max - args.domain_min) ** effective_domain_dim

        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * args.total_evolving_time)
        self.distribution_domain = Uniform(self.mins, self.maxs)

        self.test_data = self.prepare_test_data()

        # self.run_particle_method_baseline()

    def prepare_test_data(self):
        # use particle method to generate the test dataset
        # 1. sample particles from the initial distribution
        z_test = self.distribution_0.sample(self.args.batch_size_test_ref, random.PRNGKey(1234))
        if jax.devices()[0].platform == "gpu":
            z_test = jax.device_put(z_test, jax.devices("gpu")[1])
        # 2. evolve the system to t = self.total_evolving_time
        def velocity(z: jnp.ndarray):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = conv_fn_vmap(z, z)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        states_0 = {
            "z": z_test,
        }

        def ode_func1(states, t):
            dz = velocity(states["z"])

            return {"z": dz}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.args.ODE_tolerance, rtol=self.args.ODE_tolerance)
        z_T = result_forward["z"][1]
        # x_T, v_T = jnp.split(z_T, indices_or_sections=2, axis=-1)

        print(f"preparing the ground truth by running the particle method with {self.args.batch_size_test_ref} particles.")
        if jax.devices()[0].platform == "gpu":
            # x_T = jax.device_put(x_T, jax.devices("gpu")[0])
            # v_T = jax.device_put(v_T, jax.devices("gpu")[0])
            z_T = jax.device_put(z_T, jax.devices("gpu")[0])
        return {"z_T": z_T, }


    def run_particle_method_baseline(self):
        z_particle = self.distribution_0.sample(self.args.batch_size_ref, random.PRNGKey(12345))
        def velocity(z: jnp.ndarray):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = conv_fn_vmap(z, z)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        states_0 = {
            "z": z_particle,
        }

        def ode_func1(states, t):
            dz = velocity(states["z"])

            return {"z": dz}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.args.ODE_tolerance, rtol=self.args.ODE_tolerance)
        z_T = result_forward["z"][1]

        l2_error, l2 = 0, 0
        z_trues = jnp.split(self.test_data["z_T"], 10, axis=0)
        for z_true in z_trues:
            acceleration_pred = conv_fn_vmap(z_true, z_T)
            acceleration_true = self.ground_truth(z_true)
            l2_error = l2_error + jnp.sum(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
            l2 = l2 + jnp.sum(jnp.sqrt(jnp.sum((acceleration_true) ** 2, axis=-1)))

        relative_l2 = l2_error / l2
        print(f"The particle method baseline with {self.args.batch_size_ref} particles has relative l2 error {relative_l2}.")

    def ground_truth(self, zs: jnp.ndarray):
        # return the ground truth acceleration at t = self.total_evolving_time

        return conv_fn_vmap(zs, self.test_data["z_T"])


