import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import grad, vjp
import jax.numpy as jnp
from utils.common_utils import divergence_fn
from jax.experimental.ode import odeint
from utils.plot_utils import plot_scatter_2d
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from core.model import get_model
from utils.common_utils import compute_pytree_norm
import jax.random as random

def value_and_grad_fn_exact(forward_fn, params, data, rng, config, pde_instance: KineticFokkerPlanck):
    # unpack the parameters
    Gamma = pde_instance.Gamma
    beta = pde_instance.beta
    target_potential = pde_instance.target_potential
    T = pde_instance.total_evolving_time
    # unpack the data
    z_0 = data["data_initial"]
    xi_0 = pde_instance.distribution_0.score(z_0)

    params_flat, params_tree = tree_flatten(params)



    def bar_f(_z, _t, _params):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        dx = v
        dv = - Gamma * beta * forward_fn(_params, _t, _z) - beta * target_potential.gradient(x) - 4 * beta / Gamma * v
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    def f(_z, _t, _params):
        score = forward_fn(_params, _t, _z)
        return score

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    loss_0 = jnp.zeros([])
    states_0 = [z_0, xi_0, loss_0]

    def ode_func1(states, t):
        z = states[0]
        xi = states[1]
        f_t_theta = lambda _x: f(_x, t, params)
        bar_f_t_theta = lambda _x: bar_f(_x, t, params)
        dz = bar_f_t_theta(z)

        def h_t_theta(in_1, in_2):
            # in_1 is xi
            # in_2 is z
            # in_3 is theta
            div_bar_f_t_theta = lambda _z: divergence_fn(bar_f_t_theta, _z).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(bar_f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        dxi = h_t_theta(xi, z)

        def g_t(in_1, in_2):
            # in_1 is xi
            # in_2 is z
            f_t_theta_in_2 = f_t_theta(in_2)
            score_x, score_v = jnp.split(in_1, indices_or_sections=2, axis=-1)
            return jnp.mean(jnp.sum((f_t_theta_in_2 - score_v) ** 2, axis=(1,)))

        dloss = g_t(xi, z)

        return [dz, dxi, dloss]

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func1, states_0, tspace, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])
    z_T = result_forward[0][1]
    xi_T = result_forward[1][1]
    loss_f = result_forward[2][1]
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    a_T = jnp.zeros_like(z_T)
    b_T = jnp.zeros_like(z_T)
    grad_T = [jnp.zeros_like(_var) for _var in params_flat]
    loss_T = jnp.zeros([])
    states_T = [z_T, a_T, b_T, xi_T, loss_T, grad_T]

    def ode_func2(states, t):
        t = T - t
        z = states[0]
        a = states[1]
        b = states[2]
        xi = states[3]

        f_t = lambda _x, _params: f(_x, t, _params)
        bar_f_t = lambda _x, _params: bar_f(_x, t, _params)
        dx = bar_f_t(z, params)

        _, vjp_fx_fn = vjp(lambda _x: bar_f_t(_x, params), z)
        vjp_fx_a = vjp_fx_fn(a)[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(z, _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(a)[0]

        def h_t(in_1, in_2, in_3):
            # in_1 is xi
            # in_2 is z
            # in_3 is theta
            bar_f_t_theta = lambda _z: bar_f_t(_z, in_3)
            div_bar_f_t_theta = lambda _z: divergence_fn(bar_f_t_theta, _z).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(bar_f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        _, vjp_hxi_fn = vjp(lambda _xi: h_t(_xi, z, params), xi)
        vjp_hxi_b = vjp_hxi_fn(b)[0]
        _, vjp_hx_fn = vjp(lambda _x: h_t(xi, _x, params), z)
        vjp_hx_b = vjp_hx_fn(b)[0]
        _, vjp_htheta_fn = vjp(lambda _params: h_t(xi, z, _params), params)
        vjp_htheta_b = vjp_htheta_fn(b)[0]

        def g_t(in_1, in_2, in_3):
            # in_1 is xi
            # in_2 is z
            # in_3 is theta
            f_t_in_2_in_3 = f_t(in_2, in_3)
            score_x, score_v = jnp.split(in_1, indices_or_sections=2, axis=-1)
            return jnp.mean(jnp.sum((f_t_in_2_in_3 - score_v) ** 2, axis=(1,)))

        dxig = grad(g_t, argnums=0)
        dxg = grad(g_t, argnums=1)
        dthetag = grad(g_t, argnums=2)

        da = - vjp_fx_a - vjp_hx_b - dxg(xi, z, params)
        db = - vjp_hxi_b - dxig(xi, z, params)
        dxi = h_t(xi, z, params)
        dloss = g_t(xi, z, params)[None]

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        vjp_htheta_b_flat, _ = tree_flatten(vjp_htheta_b)
        dthetag_flat, _ = tree_flatten(dthetag(xi, z, params))
        dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in
                 zip(vjp_ftheta_a_flat, vjp_htheta_b_flat, dthetag_flat)]
        # dgrad = vjp_ftheta_a + vjp_htheta_b + dthetag(xi, x, params)

        return [-dx, -da, -db, -dxi, dloss, dgrad]

    # ================ Backward ==================
    tspace = jnp.array((0., T))
    result_backward = odeint(ode_func2, states_T, tspace, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])

    grad_T = tree_unflatten(params_tree, [_var[-1] for _var in result_backward[5]])
    # x_0_b = result_backward[0][-1]
    # ref_0_b = result_backward[6][-1]
    # xi_0_b = result_backward[3][-1]

    # These quantities are for the purpose of debug
    # error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
    # error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))
    # error_ref = jnp.mean(jnp.sum((ref_0 - ref_0_b).reshape(ref_0.shape[0], -1) ** 2, axis=(1,)))
    # loss_b = result_backward[4][-1]
    grad_norm = compute_pytree_norm(grad_T)
    return {
        "loss": loss_f,
        "grad": grad_T,
        "grad norm": grad_norm,
        # "ODE error x": jnp.mean(jnp.sum((result_backward["z"][-1] - states_0["z"]) ** 2, axis=-1)),
        # "ODE error ref": jnp.mean(jnp.sum((result_backward["ref"][-1] - states_0["ref"]) ** 2, axis=-1)),
    }

# choose either stochastic gradient estimator or the exact one.
value_and_grad_fn = value_and_grad_fn_exact


def plot_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    T = pde_instance.total_evolving_time
    Gamma = pde_instance.Gamma
    beta = pde_instance.beta
    target_potential = pde_instance.target_potential

    mins = pde_instance.mins
    maxs = pde_instance.maxs
    # Define the dynamics
    def bar_f(_z, _t):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        dx = v
        dv = - Gamma * beta * forward_fn(_t, _z) - beta * target_potential.gradient(x) - 4 * beta / Gamma * v
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    # sample initial data

    z_0 = pde_instance.distribution_0.sample(batch_size=1000, key=jax.random.PRNGKey(1))
    states_0 = [z_0]

    def ode_func1(states, t):
        z = states[0]
        dz = bar_f(z, t)
        return [dz]

    tspace = jnp.linspace(0, T, num=20)
    result_forward = odeint(ode_func1, states_0, tspace, atol=1e-6, rtol=1e-6)
    z_0T = result_forward[0]
    x_0T, v_0T = jnp.split(z_0T, indices_or_sections=2, axis=-1)

    plot_scatter_2d(x_0T, mins, maxs)

def test_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    Gamma = pde_instance.Gamma
    beta = pde_instance.beta
    target_potential = pde_instance.target_potential
    # compute the KL divergence and Fisher-information
    def bar_f(_z, _t):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        dx = v
        dv = - Gamma * beta * forward_fn(_t, _z) - beta * target_potential.gradient(x) - 4 * beta / Gamma * v
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    init_data = pde_instance.distribution_0.sample(batch_size=1000, key=jax.random.PRNGKey(1))
    test_time_stamps = pde_instance.test_data[0]

    data_0 = init_data
    log_density_0 = pde_instance.distribution_0.logdensity(init_data)
    score_0 = pde_instance.distribution_0.score(init_data)
    states_0 = [data_0, log_density_0, score_0]

    def ode_func(states, t):
        bar_f_t_theta = lambda _x: bar_f(_x, t)

        x = states[0]
        dx = bar_f(x, t)

        def dlog_density_func(in_1):
            # in_1 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x)
            return -div_bar_f_t_theta(in_1)

        d_logdensity = dlog_density_func(x)

        def dscore_func(in_1, in_2):
            # in_1 is score
            # in_2 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(bar_f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        _score = states[2]
        d_score = dscore_func(_score, x)

        return [dx, d_logdensity, d_score]

    tspace = test_time_stamps
    result_forward = odeint(ode_func, states_0, tspace, atol=1e-6, rtol=1e-6)

    xs = result_forward[0]  # the first axis is time, the second axis is batch, the last axis is problem dimension
    log_densities = result_forward[1]  # the first axis is time, the second axis is batch
    scores = result_forward[2]  # the first axis is time, the second axis is batch, the last axis is problem dimension

    scores_true, log_densities_true = pde_instance.ground_truth(xs)

    KL = jnp.mean(log_densities - log_densities_true, axis=(0, 1))
    Fisher_information = jnp.mean(jnp.sum((scores - scores_true) ** 2, axis=-1), axis=(0, 1))

    return {"KL": KL, "Fisher Information": Fisher_information}

def create_model_fn(pde_instance: KineticFokkerPlanck):
    # net = KiNet(time_embedding_dim=20, append_time=False)
    net = get_model(pde_instance.cfg)
    params = net.init(random.PRNGKey(11), jnp.zeros(1), pde_instance.distribution_0.sample(1, random.PRNGKey(1)))
    # params = net.init(random.PRNGKey(11), jnp.zeros(1),
    #                   jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))))
    return net, params

