from example_problems.euler_poisson_example import EulerPoisson, t_0, threshold_0
import jax.numpy as jnp
import jax
import jax.random as random

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

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
    if t.ndim == 1:
        t = t[0]
    
    if x.ndim == 2:
        return - 2 * x / 9 / (t + t_0) ** 2 - ground_truth_op_vmapx(t, x)
    elif x.ndim == 1:
        return - 2 * x / 9 / (t + t_0) ** 2 - ground_truth_op_uniform(t, x)
    else:
        raise ValueError("x should be either 1D or 2D array.")

def ground_truth_op_uniform(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
    if t.ndim == 1:
        t = t[0]

    coulomb_field_uniform = jax.grad(coulomb_potential_uniform_fn, argnums=1)
    return -coulomb_field_uniform(t, x)

ground_truth_op_vmapx = jax.vmap(ground_truth_op_uniform, in_axes=[None, 0])
ground_truth_op_vmapx_vmapt = jax.vmap(ground_truth_op_vmapx, in_axes=[0, None])

def nabla_phi_0(x: jnp.ndarray):
    if x.shape[-1] != 3:
        raise ValueError("x should be of shape [3] or [N, 3]")
    if x.ndim == 1:
        return -ground_truth_op_uniform(jnp.zeros([]), x)
    elif x.ndim == 2: # batched
        return -ground_truth_op_vmapx(jnp.zeros([]), x)
    else:
        raise NotImplementedError
    
def _mu_0(x: jnp.ndarray):
    norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1))
    conditions = [
        norm <= threshold_0,
        norm >  threshold_0
    ]
    functions = [
        jnp.ones([]),
        jnp.zeros([]),
    ]
    value = jnp.piecewise(norm, conditions, functions)
    return value / t_0

_mu_0_vmapx = jax.vmap(_mu_0, in_axes=[0])

def mu_0(x: jnp.ndarray):
    if x.shape[-1] != 3:
        raise ValueError("x should be of shape [3] or [N, 3]")
    if x.ndim == 1:
        return _mu_0(x)
    elif x.ndim == 2: # batched
        return _mu_0_vmapx(x)
    else:
        raise NotImplementedError
        
def u_0(x: jnp.ndarray):
    return x / 3 / t_0

def u_t(t: jnp.ndarray, x: jnp.ndarray):
    return x / 3 / (t_0 + t)

def mu_t(t: jnp.ndarray, x: jnp.ndarray):
    threshold_t = (3/4/jnp.pi * (t+t_0)) ** (1/3)

    norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1))
    conditions = [
        norm <= threshold_t,
        norm >  threshold_t
    ]
    functions = [
        jnp.ones([]),
        jnp.zeros([]),
    ]
    value = jnp.piecewise(norm, conditions, functions)
    return value / (t_0 + t)

def nabla_phi_t(t: jnp.ndarray, x: jnp.ndarray):
    if x.shape[-1] != 3:
        raise ValueError("x should be of shape [3] or [N, 3]")
    if x.ndim == 1:
        return -ground_truth_op_uniform(t, x)
    elif x.ndim == 2: # batched
        return -ground_truth_op_vmapx(t, x)
    else:
        raise NotImplementedError

class EulerPoissonWithDrift(EulerPoisson):
    def __init__(self, cfg, rng):
        super(EulerPoissonWithDrift, self).__init__(cfg, rng)
        self.u_0 = u_0
        self.mu_0 = mu_0
        self.nabla_phi_0 = nabla_phi_0

        self.u_t = u_t
        self.mu_t = mu_t
        self.nabla_phi_t = nabla_phi_t

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
