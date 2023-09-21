import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import vjp
import jax.numpy as jnp
from utils.common_utils import divergence_fn, compute_pytree_norm
from jax.experimental.ode import odeint
from utils.plot_utils import plot_scatter_2d
from example_problems.euler_poisson_example import EulerPoisson, conv_fn_vmap
from core.model import get_model
import jax.random as random


def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: EulerPoisson):
    # unpack the parameters
    T = pde_instance.total_evolving_time
    # unpack the data
    x_0, x_ref = data["data_initial"], data["data_ref"]
    v_0, v_ref = pde_instance.u_0(x_0), pde_instance.u_0(x_ref)

    params_flat, params_tree = tree_flatten(params)

    def hypothesis_velocity_field_fn(z, t, _params):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        dx = v
        dv = forward_fn(_params, t, x) + pde_instance.drift_term(t, x)
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    def conv_pred_fn(z, t, _params):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        return forward_fn(_params, t, x)

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    states_0 = {
        "z": jnp.concatenate([x_0, v_0], axis=-1),
        "ref": jnp.concatenate([x_ref, v_ref], axis=-1),
        "loss": jnp.zeros([]),
    }

    def ode_func1(states, t):
        def g_t(z, ref):
            x, _ = jnp.split(z, indices_or_sections=2, axis=-1)
            x_ref, _ = jnp.split(ref, indices_or_sections=2, axis=-1)
            conv_pred = conv_pred_fn(z, t, params)
            conv = conv_fn_vmap(x, x_ref)
            return jnp.mean(jnp.sum((conv_pred - conv) ** 2, axis=-1))

        return {
            "z": hypothesis_velocity_field_fn(states["z"], t, params),
            "ref": hypothesis_velocity_field_fn(states["ref"], t, params),
            "loss": g_t(states["z"], states["ref"])
        }

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func1, states_0, tspace, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])
    loss_f = result_forward["loss"][-1]
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    states_T = {
        "z": result_forward["z"][-1],
        "ref": result_forward["ref"][-1],
        "a": jnp.zeros_like(states_0["z"]),
        "b": jnp.zeros_like(states_0["ref"]),
        "grad": [jnp.zeros_like(_var) for _var in params_flat],
        "loss": jnp.zeros([])
    }


    def ode_func2(states, t):
        t = T - t

        f_t = lambda _z, _params: conv_pred_fn(_z, t, _params)
        bar_f_t = lambda _z, _params: hypothesis_velocity_field_fn(_z, t, _params)


        _, vjp_fx_fn = vjp(lambda _z: bar_f_t(_z, params), states["z"])
        vjp_fx_a = vjp_fx_fn(states["a"])[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(states["z"], _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(states["a"])[0]

        _, vjp_fxref_fn = vjp(lambda _x: bar_f_t(_x, params), states["ref"])
        vjp_fxref_b = vjp_fxref_fn(states["b"])[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(states["ref"], _params), params)
        vjp_ftheta_b = vjp_ftheta_fn(states["b"])[0]


        def g_t(z, ref, _params):
            x, _ = jnp.split(z, indices_or_sections=2, axis=-1)
            x_ref, _ =  jnp.split(ref, indices_or_sections=2, axis=-1)
            conv_pred = f_t(z, _params)
            conv = conv_fn_vmap(x, x_ref)
            return jnp.mean(jnp.sum((conv_pred - conv) ** 2, axis=-1))

        dxg = jax.grad(g_t, argnums=0)
        dxrefg = jax.grad(g_t, argnums=1)
        dthetag = jax.grad(g_t, argnums=2)

        da = - vjp_fx_a - dxg(states["z"], states["ref"], params)
        db = - vjp_fxref_b - dxrefg(states["z"], states["ref"], params)

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        vjp_ftheta_b_flat, _ = tree_flatten(vjp_ftheta_b)
        dthetag_flat, _ = tree_flatten(dthetag(states["z"], states["ref"], params))
        dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3
                 in zip(vjp_ftheta_a_flat, vjp_ftheta_b_flat, dthetag_flat)]

        return {
            "z": -bar_f_t(states["z"], params),
            "ref": -bar_f_t(states["ref"], params),
            "a": -da,
            "b": -db,
            "loss": g_t(states["z"], states["ref"], params),
            "grad": dgrad,
        }

    # ================ Backward ==================
    tspace = jnp.array((0., T))
    result_backward = odeint(ode_func2, states_T, tspace, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])
    grad = tree_unflatten(params_tree, [_var[-1] for _var in result_backward["grad"]])
    grad_norm = compute_pytree_norm(grad)

    return {
        "loss": loss_f,
        "grad": grad,
        "grad norm": grad_norm,
        "ODE error x": jnp.mean(jnp.sum((result_backward["z"][-1] - states_0["z"]) ** 2, axis=-1)),
        "ODE error ref": jnp.mean(jnp.sum((result_backward["ref"][-1] - states_0["ref"]) ** 2, axis=-1)),
    }


def plot_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    def hypothesis_velocity_field_fn(_z, _t):
        x, v = jnp.split(_z, indices_or_sections=2, axis=-1)
        dx = v
        dv = forward_fn(_t, x) + pde_instance.drift_term(_t, x)
        dz = jnp.concatenate([dx, dv], axis=-1)
        return dz

    mins = pde_instance.mins
    maxs = pde_instance.maxs

    x_0 = pde_instance.distribution_0.sample(10000, random.PRNGKey(123))
    states_0 = {"z": jnp.concatenate([x_0, pde_instance.u_0(x_0)], axis=-1)}

    def ode_func1(states, t):
        return {"z": hypothesis_velocity_field_fn(states["z"], t)}

    tspace = jnp.linspace(0, pde_instance.total_evolving_time, 11)
    result_forward = odeint(ode_func1, states_0, tspace, atol=1e-4, rtol=1e-4)
    z_0T = result_forward["z"]


def test_fn(forward_fn, config, pde_instance: EulerPoisson, rng):
    x_ground_truth = pde_instance.test_data["x_T"]
    conv_pred = forward_fn(jnp.ones(1) * pde_instance.total_evolving_time, x_ground_truth)
    conv_true = pde_instance.ground_truth(x_ground_truth)
    relative_l2 = jnp.mean(jnp.sqrt(jnp.sum((conv_pred - conv_true) ** 2, axis=-1)))
    relative_l2 = relative_l2 / jnp.mean(jnp.sqrt(jnp.sum(conv_true ** 2, axis=-1)))

    return {"relative l2 error": relative_l2}


def create_model_fn(pde_instance: EulerPoisson):
    net = get_model(pde_instance.cfg)
    # net = KiNet(output_dim=3, time_embedding_dim=0)
    # net = KiNet_Debug(output_dim=3, time_embedding_dim=0)
    # net = KiNet_Debug_2(output_dim=3, time_embedding_dim=0)
    # net = KiNet_ResNet(output_dim=3, time_embedding_dim=16, n_resblocks=3)
    params = net.init(random.PRNGKey(11), jnp.zeros(1), pde_instance.distribution_x_0.sample(1, random.PRNGKey(1)))
    return net, params

