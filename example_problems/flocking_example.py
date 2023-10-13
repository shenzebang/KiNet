import jax.numpy as jnp
from api import ProblemInstance
from core.distribution import Uniform, Gaussian, DistributionKinetic, Uniform_over_3d_Ball
import jax.random as random
import jax
from jax.experimental.ode import odeint
import gc
import warnings

Sigma_x_0 = jnp.eye(3)
mu_x_0 = - jnp.zeros(3)
distribution_x_0 = Gaussian(mu_x_0, Sigma_x_0)

Sigma_v_0 = jnp.eye(3) * 2
# mu_v_0 = jnp.array([0., 0., 0.])
mu_v_0 = jnp.zeros(3)
distribution_v_0 = Gaussian(mu_v_0, Sigma_v_0)
# =============================================

# =============================================
# Flocking Kernel in 3D!
beta = .2
def K_fn(z1: jnp.ndarray, z2: jnp.ndarray):
    dz = z1 - z2
    dx, dv = jnp.split(dz, indices_or_sections=2, axis=-1)
    dx_norm2 = jnp.sum(dx ** 2, axis=-1)
    return -dv / (1+dx_norm2) ** beta

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0])

def conv_fn(z1: jnp.ndarray, z2: jnp.ndarray):
    K = K_fn_vmapy(z1, z2)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None])
# =============================================

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    assert x.shape[-1] >= 2
    x1, x2, etc = jnp.split(x, indices_or_sections=[1, 2,], axis=-1)
    return jnp.concatenate([-x2, x1, jnp.zeros_like(etc)], axis=-1)

class Flocking(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.distribution_0 = DistributionKinetic(distribution_x=distribution_x_0, distribution_v=distribution_v_0)
        self.drift_term = self.get_drift_term()
        # domain of interest (2d dimensional box)
        # effective_domain_dim = args.domain_dim * 2  # (2d for position and velocity)
        # self.mins = args.domain_min * jnp.ones(effective_domain_dim)
        # self.maxs = args.domain_max * jnp.ones(effective_domain_dim)
        # self.domain_area = (args.domain_max - args.domain_min) ** effective_domain_dim

        # self.distribution_domain = Uniform(self.mins, self.maxs)

        self.test_data = self.prepare_test_data()

        # self.run_particle_method_baseline()


    def get_drift_term(self):
        return drift_term

    def prepare_test_data(self):
        # use particle method to generate the test dataset
        # 1. sample particles from the initial distribution
        z_test = self.distribution_0.sample(self.cfg.test.batch_size, random.PRNGKey(1234))
        if jax.devices()[0].platform == "gpu":
            z_test = jax.device_put(z_test, jax.devices("gpu")[-1])
        # 2. evolve the system to t = self.total_evolving_time
        # def velocity(z: jnp.ndarray):
        #     x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        #     dx = v
        #     dv = conv_fn_vmap(z, z)
        #     dz = jnp.concatenate([dx, dv], axis=-1)
        #     return dz

        forward_fn = lambda t, z: conv_fn_vmap(z, z)
        velocity = self.forward_fn_to_dynamics(forward_fn)

        states_0 = {
            "z": z_test,
        }

        def ode_func1(states, t):
            return {"z": velocity(t, states["z"])}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.cfg.ODE_tolerance, rtol=self.cfg.ODE_tolerance)
        z_T = result_forward["z"][-1]

        print(f"preparing the ground truth by running the particle method with {self.cfg.test.batch_size} particles.")
        if jax.devices()[0].platform == "gpu":
            # x_T = jax.device_put(x_T, jax.devices("gpu")[0])
            # v_T = jax.device_put(v_T, jax.devices("gpu")[0])
            z_T = jax.device_put(z_T, jax.devices("gpu")[0])
        return {"z_T": z_T, }


    def run_particle_method_baseline(self):
        z_particle = self.distribution_0.sample(self.cfg.baseline.batch_size, random.PRNGKey(12345))
        forward_fn = lambda t, z: conv_fn_vmap(z, z)
        velocity = self.forward_fn_to_dynamics(forward_fn)

        # def velocity(z: jnp.ndarray):
        #     x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        #     dx = v
        #     dv = conv_fn_vmap(z, z)
        #     dz = jnp.concatenate([dx, dv], axis=-1)
        #     return dz

        states_0 = {
            "z": z_particle,
        }

        def ode_func1(states, t):
            return {"z": velocity(t, states["z"])}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.cfg.ODE_tolerance, rtol=self.cfg.ODE_tolerance)
        z_T = result_forward["z"][1]

        l2_error, l2 = 0, 0
        z_trues = jnp.split(self.test_data["z_T"], 10, axis=0)
        for z_true in z_trues:
            acceleration_pred = conv_fn_vmap(z_true, z_T)
            acceleration_true = self.ground_truth(z_true)
            l2_error = l2_error + jnp.sum(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
            l2 = l2 + jnp.sum(jnp.sqrt(jnp.sum((acceleration_true) ** 2, axis=-1)))

        relative_l2 = l2_error / l2
        print(f"The particle method baseline with {self.cfg.baseline.batch_size} particles has relative l2 error {relative_l2}.")

    def ground_truth(self, zs: jnp.ndarray):
        # return the ground truth acceleration at t = self.total_evolving_time

        return conv_fn_vmap(zs, self.test_data["z_T"])

    def forward_fn_to_dynamics(self, forward_fn):
        def dynamics(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = forward_fn(t, z) + self.drift_term(t, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        return dynamics



