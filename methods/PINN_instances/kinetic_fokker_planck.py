import jax
import functools
from jax import jacrev
import jax.numpy as jnp
from example_problems.kinetic_fokker_planck_example import beta, Gamma, test_data
from utils.plot_utils import plot_density_2d
from utils.test_utils import v_gaussian_log_density, v_gaussian_score
from core.distribution import Uniform


def value_and_grad_fn(forward_fn, params, data, key, config):
    weights = config["weights"]
    # diffusion_coefficient = config["diffusion_coefficient"]
    diffusion_coefficient = beta * Gamma
    # unpack data
    time_initial,   space_initial,  target_initial  = data["data_initial"]
    # time_boundary,  space_boundary, target_boundary = data["data_boundary"]
    time_train,     space_train,                    = data["data_train"]

    def model_loss(_params):
        def velocity(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = -beta * x - 4 * beta / Gamma * v
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        def div_velocity(t, z):
            return -4. * beta / Gamma * z.shape[-1] / 2.


        @functools.partial(jax.vmap, in_axes=(0, 0))
        def fokker_planck_eq(t, z):
            u = forward_fn(_params, t, z)
            u_t = jacrev(forward_fn, argnums=1,)(_params, t, z)
            u_z_fn = jax.jacrev(forward_fn, argnums=2)
            u_z = u_z_fn(_params, t, z)
            jacobian_fn = jax.jacfwd(u_z_fn, argnums=2)
            jacobian_diag = jnp.diag(jacobian_fn(_params, t, z)[0])
            jacobian_diag_x, jacobian_diag_v = jnp.split(jacobian_diag, indices_or_sections=2, axis=-1)
            u_laplacian_v = jnp.sum(jacobian_diag_v)
            return u_t + div_velocity(t, z) * u + jnp.dot(u_z, velocity(t, z)) \
                   - u_laplacian_v * diffusion_coefficient

        # fokker_planck_eq = jax.vmap(fokker_planck_eq, in_axes=(0, None))
        # fokker_planck_eq = jax.vmap(fokker_planck_eq, in_axes=(None, 0))


        # def change_of_mass_t(t, z):
        #     u_t_fn = jacrev(forward_fn, argnums=1, )
        #     u_t_fn = jax.vmap(u_t_fn, in_axes=[None, None, 0])
        #     u_t = u_t_fn(_params, t, z)
        #     return jnp.mean(u_t)
        #
        # change_of_mass = jax.vmap(change_of_mass_t, in_axes=[0, None])
        # def mass_preservation(t, z):
        #     mass_change = change_of_mass(t, z)
        #     return jnp.mean(mass_change ** 2)




        # @functools.partial(jax.vmap, in_axes=(None, 0, 0))
        # def fokker_planck_eq(__params, t, x):
        #     u = forward_fn(__params, t, x)
        #     u_t = jacrev(forward_fn, argnums=1,)(__params, t, x)
        #     u_x_fn = jax.jacrev(forward_fn, argnums=2)
        #     u_x = u_x_fn(__params, t, x)
        #     jacobian_fn = jax.jacfwd(u_x_fn, argnums=2)
        #     u_laplacian = jnp.sum(jnp.diag(jacobian_fn(__params, t, x)[0]))
        #     return u_t - fokker_planck_example.tr_inv_Cov_inf * u \
        #            - jnp.dot(u_x, jnp.matmul(fokker_planck_example.inv_Cov_inf, x - fokker_planck_example.mu_inf)) \
        #            - u_laplacian

        # u_pred_boundary = forward_fn(_params, time_boundary,    space_boundary)
        u_pred_initial  = forward_fn(_params, time_initial, space_initial)
        f_pred_train    = fokker_planck_eq(time_train, space_train)
        # loss_u_boundary = jnp.mean((u_pred_boundary - target_boundary) ** 2)
        loss_u_initial  = jnp.mean((u_pred_initial - target_initial[:, None]) ** 2)
        loss_f_train    = jnp.mean((f_pred_train) ** 2)

        # return loss_u_boundary * weights["weight_boundary"] + loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]

        return loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]

    v_g = jax.value_and_grad(model_loss)
    value, grad = v_g(params)
    return value, grad

def test_density_op(density_fn, config):
    mins = config["mins"]
    maxs = config["maxs"]
    domain_area = config["domain_area"]

    # These functions take a single point (t, x) as input
    rho = density_fn
    log_rho = lambda t, x: jnp.log(jnp.maximum(density_fn(t, x), 1e-8))
    nabla_log_rho = jax.jacrev(log_rho, argnums=1)

    # unpack the test data
    test_time_stamps, mus, Sigmas = test_data

    # side_x = jnp.linspace(mins[0], maxs[0], 256)
    # side_y = jnp.linspace(mins[1], maxs[1], 256)
    # X, Y = jnp.meshgrid(side_x, side_y)
    # grid_points_test = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    # grid_points_test = jnp.concatenate([grid_points_test, grid_points_test], axis=-1)
    distribution_0 = Uniform(mins, maxs)
    points_test = distribution_0.sample(256*256, config["key"])

    rho = jax.vmap(jax.vmap(rho, in_axes=[None, 0]), in_axes=[0, None])
    log_rho = jax.vmap(jax.vmap(log_rho, in_axes=[None, 0]), in_axes=[0, None])
    nabla_log_rho = jax.vmap(jax.vmap(nabla_log_rho, in_axes=[None, 0]), in_axes=[0, None])

    densities = jnp.maximum(rho(test_time_stamps[:, None], points_test), 1e-10)
    log_densities = log_rho(test_time_stamps[:, None], points_test)
    scores = jnp.squeeze(nabla_log_rho(test_time_stamps[:, None], points_test))

    v_v_gaussian_log_density = jax.vmap(v_gaussian_log_density, in_axes=[None, 0, 0])
    log_densities_true = v_v_gaussian_log_density(points_test, Sigmas, mus)[:, :, None]

    v_v_gaussian_score = jax.vmap(v_gaussian_score, in_axes=[None, 0, 0])
    scores_true = v_v_gaussian_score(points_test, Sigmas, mus)

    KL = jnp.mean(densities * (log_densities - log_densities_true)) * domain_area
    L1 = jnp.mean(jnp.abs(densities - jnp.exp(log_densities_true))) * domain_area
    total_mass = jnp.mean(densities) * domain_area
    total_mass_true = jnp.mean(jnp.exp(log_densities_true)) * domain_area

    Fisher_information = jnp.mean(jnp.sum((scores - scores_true) ** 2, axis=(-1)), axis=(0, 1))

    # print(f"KL {KL: .2f}, L1 {L1: .2f}, Fisher information {Fisher_information: .2f}")
    # print(f"Total mass {total_mass: .2f}, True total mass {total_mass_true: .2f}")
    return L1, KL

# def plot_density_op(density_fn, config):
#     T = config["T"]
#     t_part = config["t_part"]
#     for t in range(t_part):
#         def f(x: jnp.ndarray):
#             batch_size_x = x.shape[0]
#             return density_fn(jnp.ones((batch_size_x, 1)) * T/t_part * t, x)
#
#         plot_density_2d(f, config)