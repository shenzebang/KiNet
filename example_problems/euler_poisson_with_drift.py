from example_problems.euler_poisson_example import EulerPoisson, t_0
import jax.numpy as jnp
import jax
import jax.random as random

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    t = t + t_0
    return - 2 * x / 9 / t ** 2 - x / 3 / t

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
        1 / 8 /jnp.pi / xi_norm
    ]
    return jnp.piecewise(xi_norm, conditions, functions)

def ground_truth_op_uniform(t: jnp.ndarray, x: jnp.ndarray):
    coulomb_field_uniform = jax.grad(coulomb_potential_uniform_fn, argnums=1)
    return -coulomb_field_uniform(t[0], x)

ground_truth_op_vmapx = jax.vmap(ground_truth_op_uniform, in_axes=[None, 0])

class EulerPoissonWithDrift(EulerPoisson):
    def __init__(self, args, rng):
        super(EulerPoissonWithDrift, self).__init__(args, rng)

    def get_drift_term(self):
        return drift_term

    def prepare_test_data(self):
        print(f"Using the instance {self.instance_name}. Will use the close-form solution to test accuracy.")
        x_test = self.distribution_0.sample(self.args.batch_size_test_ref, random.PRNGKey(1234))
        return {"x_T": x_test, }

    def ground_truth(self, xs: jnp.ndarray):
        return ground_truth_op_vmapx(self.total_evolving_time, xs)
