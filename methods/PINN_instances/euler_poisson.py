import jax
import functools
from jax import jacrev
import jax.numpy as jnp
from utils.plot_utils import plot_density_2d
from core.distribution import Uniform
from core.model import MLP
from example_problems.euler_poisson_example import EulerPoisson
from core.normalizing_flow import RealNVP, MNF
import jax.random as random
from functools import partial
from flax import linen as nn
from typing import List

def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: EulerPoisson):
    weights = config["weights"]
    # unpack data
    time_initial, space_initial, target_initial = data["data_initial"]
    time_train, space_train, = data["data_train"]

    def model_loss(_params):
        mu_fn = lambda t, z: forward_fn(_params, t, z, ["mu"])["mu"]
        u_fn = lambda t, z: forward_fn(_params, t, z, ["u"])["u"]
        nabla_phi_fn = lambda t, z: forward_fn(_params, t, z, ["nabla_phi"])["nabla_phi"]
        # ====================================================================================================
        # residual
        def euler_poisson_equation(t, z):
            mu_t = jacrev(mu_fn, argnums=0,)(t, z)[0]

            u_mu_fn = lambda t, z: mu_fn(t, z) * u_fn(t, z)
            jac_u_mu_fn = jax.jacfwd(u_mu_fn, argnums=1)
            div_u_mu = jnp.sum(jnp.diag(jac_u_mu_fn(t, z)))
            term_1 = (mu_t + div_u_mu) ** 2

            u_t = jacrev(u_fn, argnums=0, )(t, z)[0]
            uu_fn = lambda _z: jnp.dot(u_fn(t, _z), u_fn(t, z))
            u_nabla_u_fn = jax.grad(uu_fn)
            term_2 = jnp.sum((u_t + u_nabla_u_fn(z) + nabla_phi_fn(t, z))**2, axis=-1)

            jac_nabla_phi_fn = jax.jacfwd(nabla_phi_fn, argnums=1)
            laplacian_phi = jnp.sum(jnp.diag(jac_nabla_phi_fn(t, z)))
            term_3 = jnp.sum((laplacian_phi + mu_fn(t, z)) ** 2, axis=-1)

            return term_1 + term_2 + term_3

        euler_poisson_equation = jax.vmap(jax.vmap(euler_poisson_equation, in_axes=(None, 0)), in_axes=(0, None))
        residual = euler_poisson_equation(time_train, space_train)
        # ====================================================================================================

        # ====================================================================================================
        # mass change
        def mass_change(t, z):
            u_t = jacrev(mu_fn, argnums=0,)(t, z)[0]
            return u_t

        mass_change = jax.vmap(mass_change, in_axes=(None, 0))
        def mass_change_t(t, z):
            return jnp.mean(mass_change(t, z))

        mass_change_t = jax.vmap(mass_change_t, in_axes=(0, None))
        mass_change_total = mass_change_t(time_train, space_train)
        # ====================================================================================================

        # ====================================================================================================
        # initial loss
        vv_mu_fn = jax.vmap(jax.vmap(mu_fn, in_axes=(None, 0)), in_axes=(0, None))
        mu_pred_initial = vv_mu_fn(time_initial, space_initial)
        loss_mu_initial = jnp.mean((mu_pred_initial - target_initial[:, None]) ** 2)

        vv_u_fn = jax.vmap(jax.vmap(u_fn, in_axes=(None, 0)), in_axes=(0, None))
        u_pred_initial = vv_u_fn(time_initial, space_initial)
        loss_u_initial = jnp.mean((u_pred_initial - pde_instance.u_0(space_initial)[:, None]) ** 2)
        # ====================================================================================================

        # ====================================================================================================
        # total loss = (loss of initial condition) + (loss of residual) + (loss of mass change)
        loss_initial = loss_u_initial + loss_mu_initial
        loss_residual = jnp.mean(residual ** 2)
        loss_mass_change = jnp.mean(mass_change_total ** 2)
        # ====================================================================================================

        return loss_initial * weights["weight_initial"] + loss_residual * weights["weight_train"]\
                + loss_mass_change * weights["mass_change"]

    v_g = jax.value_and_grad(model_loss)
    value, grad = v_g(params)
    return value, grad


def test_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    nabla_phi_fn = lambda t, z: forward_fn(t, z, ["nabla_phi"])["nabla_phi"]
    nabla_phi_fn = jax.vmap(nabla_phi_fn, in_axes=[None, 0])
    x_ground_truth = pde_instance.test_data["x_T"]
    acceleration_pred = - nabla_phi_fn(jnp.ones(1) * pde_instance.total_evolving_time, x_ground_truth)
    acceleration_true = pde_instance.ground_truth(x_ground_truth)
    relative_l2 = jnp.mean(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
    relative_l2 = relative_l2 / jnp.mean(jnp.sqrt(jnp.sum((acceleration_true) ** 2, axis=-1)))

    return {"relative l2 error": relative_l2}
    # mins = pde_instance.mins
    # maxs = pde_instance.maxs
    # domain_area = pde_instance.domain_area
    #
    # # These functions take a single point (t, x) as input
    # rho = forward_fn
    # log_rho = lambda t, x: jnp.maximum(jnp.log(forward_fn(t, x)), -100)
    # nabla_log_rho = jax.jacrev(log_rho, argnums=1)
    #
    # # unpack the test data
    # test_time_stamps = pde_instance.test_data[0]
    #
    # # side_x = jnp.linspace(mins[0], maxs[0], 256)
    # # side_y = jnp.linspace(mins[1], maxs[1], 256)
    # # X, Y = jnp.meshgrid(side_x, side_y)
    # # grid_points_test = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    # # grid_points_test = jnp.concatenate([grid_points_test, grid_points_test], axis=-1)
    # distribution_0 = Uniform(mins, maxs)
    # points_test = distribution_0.sample(256 * 256, rng)
    #
    # rho = jax.vmap(jax.vmap(rho, in_axes=[None, 0]), in_axes=[0, None])
    # log_rho = jax.vmap(jax.vmap(log_rho, in_axes=[None, 0]), in_axes=[0, None])
    # nabla_log_rho = jax.vmap(jax.vmap(nabla_log_rho, in_axes=[None, 0]), in_axes=[0, None])
    #
    # densities = rho(test_time_stamps[:, None], points_test)
    # log_densities = log_rho(test_time_stamps[:, None], points_test)
    # scores = jnp.squeeze(nabla_log_rho(test_time_stamps[:, None], points_test))
    #
    # scores_true, log_densities_true = pde_instance.ground_truth(points_test)
    #
    # densities = jnp.squeeze(densities)
    # log_densities = jnp.squeeze(log_densities)
    #
    #
    # KL = jnp.mean(densities * (log_densities - log_densities_true)) * domain_area
    # L1 = jnp.mean(jnp.abs(densities - jnp.exp(log_densities_true))) * domain_area
    # total_mass = jnp.mean(densities) * domain_area
    # total_mass_true = jnp.mean(jnp.exp(log_densities_true)) * domain_area
    #
    # Fisher_information = jnp.mean(jnp.sum((scores - scores_true) ** 2, axis=-1))
    #
    # # print(f"KL {KL: .2f}, L1 {L1: .2f}, Fisher information {Fisher_information: .2f}")
    # # print(f"Total mass {total_mass: .2f}, True total mass {total_mass_true: .2f}")
    # return {"L1": L1, "KL": KL, "Fisher Information": Fisher_information, "total_mass": total_mass, "total_mass_true": total_mass_true}
    # return {}


def plot_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    pass
    # T = KineticFokkerPlanck.total_evolving_time
    # t_part = config["t_part"]
    # for t in range(t_part):
    #     def f(x: jnp.ndarray):
    #         batch_size_x = x.shape[0]
    #         return forward_fn(jnp.ones((batch_size_x, 1)) * T / t_part * t, x)
    #
    #     plot_density_2d(f, config)


class MLPEulerPoisson(nn.Module):
    scaling: float = 1.
    def setup(self):
        self.mu = MLP(output_dim=1)
        self.u = MLP(output_dim=3)
        self.nabla_phi = MLP(output_dim=3)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, keys: List[str]):
        result = {}
        for key in keys:
            if key == "mu":
                result[key] = self.mu(t, x) * self.scaling
            elif key == "u":
                result[key] = self.u(t, x)
            elif key == "nabla_phi":
                result[key] = self.nabla_phi(t, x)
            else:
                raise Exception("(PINN) unknown key!")
        return result

def create_model_fn(problem_instance: EulerPoisson):

    net = MLPEulerPoisson()

    params = net.init(random.PRNGKey(11), jnp.zeros(1),
                      jnp.squeeze(problem_instance.distribution_0.sample(1, random.PRNGKey(1))), ["mu", "u", "nabla_phi"])

    # set the scaling of mu so that the total mass is of the right order.
    mins = problem_instance.mins
    maxs = problem_instance.maxs
    domain_area = problem_instance.domain_area
    rho = partial(net.apply, params)
    rho = jax.vmap(rho, in_axes=[None, 0, None])
    distribution_0 = Uniform(mins, maxs)
    points_test = distribution_0.sample(256 * 256, random.PRNGKey(123))
    densities = rho(jnp.zeros(1), points_test, ["mu"])["mu"]
    total_mass = jnp.mean(densities) * domain_area

    net.scaling = 1./ total_mass

    print(
        f"(PINN) Automatically set the scaling in mu to 1/{total_mass: .2e}."
    )

    return net, params

# def create_model_fn(log_prob_0):
#     param_dict = {
#         'dim': 4,
#         'embed_time_dim': 0,
#         'couple_mul': 2,
#         'mask_type': 'loop',
#         'activation_layer': 'celu',
#         'soft_init': 0.,
#         'ignore_time': False,
#     }
#     mnf = MNF(**param_dict)
#     return RealNVP(mnf, log_prob_0)