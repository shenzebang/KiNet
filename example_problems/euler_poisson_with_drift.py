from example_problems.euler_poisson_example import EulerPoisson, t_0
import jax.numpy as jnp
import jax
import jax.random as random

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    t = t + t_0
    return - 2 * x / 9 / t ** 2 - x / 3 / t

def u_0(x: jnp.ndarray):
    return x / 3 / t_0

def coulomb_potential_uniform_fn(t: jnp.ndarray, xi: jnp.ndarray):
    xi_norm_2 = jnp.sum(xi ** 2)
    xi_norm = jnp.sqrt(xi_norm_2)
    threshold_t = (3/4/jnp.pi * (t+t_0)) ** (1/3)
    conditions = [
        xi_norm <= threshold_t,
        xi_norm > threshold_t
    ]
    functions = [
        (2 * threshold_t ** 2 - xi_norm_2)/6/(t+t_0),
        1 / 4 /jnp.pi / xi_norm
    ]
    return jnp.piecewise(xi_norm, conditions, functions)

def ground_truth_op_uniform(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
    if t.ndim == 1:
        t = t[0]

    coulomb_field_uniform = jax.grad(coulomb_potential_uniform_fn, argnums=1)
    return -coulomb_field_uniform(t, x)

ground_truth_op_vmapx = jax.vmap(ground_truth_op_uniform, in_axes=[None, 0])
ground_truth_op_vmapx_vmapt = jax.vmap(ground_truth_op_vmapx, in_axes=[0, None])

class EulerPoissonWithDrift(EulerPoisson):
    def __init__(self, cfg, rng):
        super(EulerPoissonWithDrift, self).__init__(cfg, rng)
        self.u_0 = u_0

    def get_drift_term(self):
        return drift_term

    def prepare_test_data(self):
        print(f"Using the instance {self.instance_name}. Will use the close-form solution to test accuracy.")
        x_test = self.distribution_0.sample(self.cfg.test.batch_size, random.PRNGKey(1234))
        return {"x_T": x_test, }

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        assert ts.ndim == 0 or ts.ndim == 1
        if ts.ndim == 0:
            ts = ts * jnp.ones(1)
        
        return ground_truth_op_vmapx_vmapt(ts, xs)
        

    # def ground_truth_t(self, xs: jnp.ndarray, t: jnp.ndarray):
    def forward_fn_to_dynamics(self, forward_fn):
        def dynamics(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = forward_fn(t, x) + self.drift_term(t, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        return dynamics
