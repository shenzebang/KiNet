import jax
import functools
from jax import jacrev
import jax.numpy as jnp
from utils.plot_utils import plot_density_2d
from core.distribution import Uniform
from core.model import MLP
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from core.normalizing_flow import RealNVP, MNF

def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: KineticFokkerPlanck):
    weights = config["weights"]
    beta = pde_instance.beta
    Gamma = pde_instance.Gamma
    diffusion_coefficient = pde_instance.beta * pde_instance.Gamma
    # unpack data
    time_initial, space_initial, target_initial = data["data_initial"]
    time_train, space_train, = data["data_train"]

    def model_loss(_params):
        def velocity(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = -beta * x - 4 * beta / Gamma * v
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        def div_velocity(t, z):
            return -4. * beta / Gamma * z.shape[-1] / 2.

        # @functools.partial(jax.vmap, in_axes=(0, 0))
        def fokker_planck_eq(t, z):
            u = forward_fn(_params, t, z)
            u_t = jacrev(forward_fn, argnums=1, )(_params, t, z)[0]
            u_z_fn = jax.jacrev(forward_fn, argnums=2)
            u_z = u_z_fn(_params, t, z)[0]
            jacobian_fn = jax.hessian(forward_fn, argnums=2)
            jacobian_diag = jnp.diag(jacobian_fn(_params, t, z)[0])
            jacobian_diag_x, jacobian_diag_v = jnp.split(jacobian_diag, indices_or_sections=2, axis=-1)
            u_laplacian_v = jnp.sum(jacobian_diag_v)
            return u_t + div_velocity(t, z) * u + jnp.dot(u_z, velocity(t, z)) - u_laplacian_v * diffusion_coefficient

        fokker_planck_eq = jax.vmap(jax.vmap(fokker_planck_eq, in_axes=(None, 0)), in_axes=(0, None))

        def mass_change(t, z):
            u_t = jacrev(forward_fn, argnums=1, )(_params, t, z)[0]
            return u_t

        mass_change = jax.vmap(mass_change, in_axes=(None, 0))
        def mass_change_t(t, z):
            return jnp.mean(mass_change(t, z))

        mass_change_t = jax.vmap(mass_change_t, in_axes=(0, None))

        # u_pred_boundary = forward_fn(_params, time_boundary,    space_boundary)
        vv_forward = jax.vmap(jax.vmap(forward_fn, in_axes=(None, None, 0)), in_axes=(None, 0, None))
        u_pred_initial = vv_forward(_params, time_initial, space_initial)
        f_pred_train = fokker_planck_eq(time_train, space_train)
        mass_change_total = mass_change_t(time_train, space_train)
        # loss_u_boundary = jnp.mean((u_pred_boundary - target_boundary) ** 2)
        loss_u_initial = jnp.mean((u_pred_initial - target_initial[:, None]) ** 2)
        loss_f_train = jnp.mean((f_pred_train) ** 2)
        loss_mass_change = jnp.mean(mass_change_total ** 2)

        # return loss_u_boundary * weights["weight_boundary"] + loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]

        return loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]\
                + loss_mass_change * weights["mass_change"]

    v_g = jax.value_and_grad(model_loss)
    value, grad = v_g(params)
    return value, grad


def test_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    mins = pde_instance.mins
    maxs = pde_instance.maxs
    domain_area = pde_instance.domain_area

    # These functions take a single point (t, x) as input
    rho = forward_fn
    log_rho = lambda t, x: jnp.maximum(jnp.log(forward_fn(t, x)), -100)
    nabla_log_rho = jax.jacrev(log_rho, argnums=1)

    # unpack the test data
    test_time_stamps = pde_instance.test_data[0]

    # side_x = jnp.linspace(mins[0], maxs[0], 256)
    # side_y = jnp.linspace(mins[1], maxs[1], 256)
    # X, Y = jnp.meshgrid(side_x, side_y)
    # grid_points_test = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    # grid_points_test = jnp.concatenate([grid_points_test, grid_points_test], axis=-1)
    distribution_0 = Uniform(mins, maxs)
    points_test = distribution_0.sample(256 * 256, rng)

    rho = jax.vmap(jax.vmap(rho, in_axes=[None, 0]), in_axes=[0, None])
    log_rho = jax.vmap(jax.vmap(log_rho, in_axes=[None, 0]), in_axes=[0, None])
    nabla_log_rho = jax.vmap(jax.vmap(nabla_log_rho, in_axes=[None, 0]), in_axes=[0, None])

    densities = rho(test_time_stamps[:, None], points_test)
    log_densities = log_rho(test_time_stamps[:, None], points_test)
    scores = jnp.squeeze(nabla_log_rho(test_time_stamps[:, None], points_test))

    scores_true, log_densities_true = pde_instance.ground_truth(points_test)

    densities = jnp.squeeze(densities)
    log_densities = jnp.squeeze(log_densities)


    KL = jnp.mean(densities * (log_densities - log_densities_true)) * domain_area
    L1 = jnp.mean(jnp.abs(densities - jnp.exp(log_densities_true))) * domain_area
    total_mass = jnp.mean(densities) * domain_area
    total_mass_true = jnp.mean(jnp.exp(log_densities_true)) * domain_area

    Fisher_information = jnp.mean(jnp.sum((scores - scores_true) ** 2, axis=-1))

    # print(f"KL {KL: .2f}, L1 {L1: .2f}, Fisher information {Fisher_information: .2f}")
    # print(f"Total mass {total_mass: .2f}, True total mass {total_mass_true: .2f}")
    return {"L1": L1, "KL": KL, "Fisher Information": Fisher_information, "total_mass": total_mass, "total_mass_true": total_mass_true}


def plot_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    T = KineticFokkerPlanck.total_evolving_time
    t_part = config["t_part"]
    for t in range(t_part):
        def f(x: jnp.ndarray):
            batch_size_x = x.shape[0]
            return forward_fn(jnp.ones((batch_size_x, 1)) * T / t_part * t, x)

        plot_density_2d(f, config)


# def create_model_fn():
#     return MLP()

def create_model_fn(log_prob_0):
    param_dict = {
        'dim': 4,
        'embed_time_dim': 0,
        'couple_mul': 2,
        'mask_type': 'loop',
        'activation_layer': 'celu',
        'soft_init': 0.,
        'ignore_time': False,
    }
    mnf = MNF(**param_dict)
    return RealNVP(mnf, log_prob_0)