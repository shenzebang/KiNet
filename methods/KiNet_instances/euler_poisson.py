import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import grad, vjp
import jax.numpy as jnp
from utils.common_utils import divergence_fn
from jax.experimental.ode import odeint
from utils.plot_utils import plot_scatter_2d
from example_problems.euler_poisson_example import EulerPoisson, conv_fn_vmap
from core.model import KiNet
import jax.random as random


def value_and_grad_fn_exact(forward_fn, params, data, rng, config, pde_instance: EulerPoisson):
    # unpack the parameters
    ODE_tolerance = config["ODE_tolerance"]
    T = pde_instance.total_evolving_time
    # unpack the data
    x_0, x_ref = data["data_initial"], data["data_ref"]
    v_0, v_ref = pde_instance.u_0(x_0), pde_instance.u_0(x_ref)
    z_0 = jnp.concatenate([x_0, v_0], axis=-1)
    z_ref = jnp.concatenate([x_ref, v_ref], axis=-1)

    params_flat, params_tree = tree_flatten(params)

    def bar_f(_z, _t, _params):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        dx = v
        dv = forward_fn(_params, _t, x) + pde_instance.drift_term(_t, x)
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    def f(_z, _t, _params):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        return forward_fn(_params, _t, x)

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    loss_0 = jnp.zeros([])
    states_0 = [z_0, z_ref, loss_0]

    def ode_func1(states, t):
        z = states[0]
        z_ref = states[1]
        dz = bar_f(z, t, params)
        dz_ref = bar_f(z_ref, t, params)

        def g_t(_z):
            conv_pred = f(_z, t, params)
            _x, _ = jnp.split(_z, indices_or_sections=2, axis=-1)
            _x_ref, _ = jnp.split(z_ref, indices_or_sections=2, axis=-1)
            return jnp.mean(jnp.sum((conv_pred - conv_fn_vmap(_x, _x_ref)) ** 2, axis=-1))

        dloss = g_t(z)

        return [dz, dz_ref, dloss]

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func1, states_0, tspace, atol=ODE_tolerance, rtol=ODE_tolerance)
    z_T = result_forward[0][1]
    z_ref_T = result_forward[1][1]
    loss_f = result_forward[2][1]
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    a_T = jnp.zeros_like(z_T)
    b_T = jnp.zeros_like(z_ref_T)
    grad_T = [jnp.zeros_like(_var) for _var in params_flat]
    loss_T = jnp.zeros([])
    states_T = [z_T, z_ref_T, a_T, b_T, loss_T, grad_T]

    def ode_func2(states, t):
        t = T - t
        z = states[0]
        z_ref = states[1]
        a = states[2]
        b = states[3]

        f_t = lambda _x, _params: f(_x, t, _params)
        bar_f_t = lambda _x, _params: bar_f(_x, t, _params)
        dz = bar_f_t(z, params)
        dz_ref = bar_f_t(z_ref, params)


        _, vjp_fx_fn = vjp(lambda _x: bar_f_t(_x, params), z)
        vjp_fx_a = vjp_fx_fn(a)[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(z, _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(a)[0]

        _, vjp_fxref_fn = vjp(lambda _x: bar_f_t(_x, params), z_ref)
        vjp_fxref_b = vjp_fxref_fn(b)[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(z_ref, _params), params)
        vjp_ftheta_b = vjp_ftheta_fn(b)[0]


        def g_t(_z, _z_ref, _params):
            conv_pred = f_t(_z, _params)
            _x, _ = jnp.split(_z, indices_or_sections=2, axis=-1)
            _x_ref, _ =  jnp.split(_z_ref, indices_or_sections=2, axis=-1)

            return jnp.mean(jnp.sum((conv_pred - conv_fn_vmap(_x, _x_ref)) ** 2, axis=-1))

        dxg = grad(g_t, argnums=0)
        dxrefg = grad(g_t, argnums=1)
        dthetag = grad(g_t, argnums=2)

        da = - vjp_fx_a - dxg(z, z_ref, params)
        db = - vjp_fxref_b - dxrefg(z, z_ref, params)
        dloss = g_t(z, z_ref, params)

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        vjp_ftheta_b_flat, _ = tree_flatten(vjp_ftheta_b)
        dthetag_flat, _ = tree_flatten(dthetag(z, z_ref, params))
        dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3
                 in zip(vjp_ftheta_a_flat, vjp_ftheta_b_flat, dthetag_flat)]

        return [-dz, -dz_ref, -da, -db, dloss, dgrad]

    # ================ Backward ==================
    tspace = jnp.array((0., T))
    result_backward = odeint(ode_func2, states_T, tspace, atol=ODE_tolerance, rtol=ODE_tolerance)

    grad_T = tree_unflatten(params_tree, [_var[-1] for _var in result_backward[5]])
    # x_0_b = result_backward[0][-1]
    # ref_0_b = result_backward[6][-1]
    # xi_0_b = result_backward[3][-1]

    # These quantities are for the purpose of debug
    # error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
    # error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))
    # error_ref = jnp.mean(jnp.sum((ref_0 - ref_0_b).reshape(ref_0.shape[0], -1) ** 2, axis=(1,)))
    # loss_b = result_backward[4][-1]

    return loss_f, grad_T


# choose either stochastic gradient estimator or the exact one.
value_and_grad_fn = value_and_grad_fn_exact


def plot_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    pass


def test_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    x_ground_truth = pde_instance.test_data["x_T"]
    conv_pred = forward_fn(jnp.ones(1) * pde_instance.total_evolving_time, x_ground_truth)
    conv_true = pde_instance.ground_truth(x_ground_truth)
    relative_l2 = jnp.mean(jnp.sqrt(jnp.sum((conv_pred - conv_true) ** 2, axis=-1)))
    relative_l2 = relative_l2 / jnp.mean(jnp.sqrt(jnp.sum(conv_true ** 2, axis=-1)))

    return {"relative l2 error": relative_l2}


def create_model_fn(pde_instance: EulerPoisson):
    net = KiNet(output_dim=3, time_embedding_dim=0, append_time=True)
    params = net.init(random.PRNGKey(11), jnp.zeros(1),
                      jnp.squeeze(pde_instance.distribution_x_0.sample(1, random.PRNGKey(1))))
    return net, params

