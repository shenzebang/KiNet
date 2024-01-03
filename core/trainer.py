import optax
import wandb
import jax
import jax.numpy as jnp
from utils.common_utils import compute_pytree_norm
from api import Method
import jax.random as random
from optax import GradientTransformation
import copy

class JaxTrainer:
    def __init__(self,
                 cfg,
                 method: Method,
                 rng: jnp.ndarray,
                 optimizer: GradientTransformation,
                 forward_fn,
                 params: optax.Params,
                 ):
        self.cfg = cfg
        self.forward_fn = forward_fn
        self.params_initial = params
        self.optimizer = optimizer
        self.method = method
        self.rng = rng
        self.params = {"current": copy.deepcopy(params), "previous": []}
        self.time_per_shard = self.cfg.pde_instance.total_evolving_time / self.cfg.train.number_of_time_shard
        self.time_interval = {"current": jnp.array([0, self.time_per_shard]), "previous": []}

    def fit(self, ):
        # jit or pmap the gradient computation for efficiency
        def _value_and_grad_fn(params, time_interval, rng):
            return self.method.value_and_grad_fn(self.forward_fn, params, time_interval, rng)

        if self.cfg.backend.use_pmap_train and jax.local_device_count() > 1:
            _value_and_grad_fn = jax.pmap(_value_and_grad_fn, in_axes=(None, None, 0))

            def value_and_grad_fn(params, time_interval, rng):

                rngs = random.split(rng, jax.local_device_count())
                # compute in parallel
                v_g_etc = _value_and_grad_fn(params, time_interval, rngs)
                v_g_etc = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), v_g_etc)
                return v_g_etc
        else:
            value_and_grad_fn = jax.jit(_value_and_grad_fn)
            # value_and_grad_fn_efficient = value_and_grad_fn

        @jax.jit
        def step_fn(params, opt_state, grad):
            updates, opt_state = self.optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        # @jax.jit
        if self.cfg.backend.use_pmap_test and jax.local_device_count() > 1:
            def _test(params, time_interval, rng):
                return self.method.test_fn(self.forward_fn, params, time_interval, rng)
            
            _test_fn = jax.pmap(_test, in_axes=(None, None, 0))

            def test_fn(params, time_interval, rng):
                rngs = random.split(rng, jax.local_device_count())
                test_results = _test_fn(params, time_interval, rngs)
                test_results = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), test_results)
                return test_results
        else:
            def test_fn(params, time_interval, rng):
                return self.method.test_fn(self.forward_fn, params, time_interval, rng)
            
        test_fn = jax.jit(test_fn)

        # @jax.jit
        def plot_fn(params, time_interval, rng):
            return self.method.plot_fn(self.forward_fn, params, time_interval, rng)
        
        minimum_loss_collection = []
        rngs_shard = jax.random.split(self.rng, self.cfg.train.number_of_time_shard)
        for shard_id, rng_shard in enumerate(rngs_shard):
            minimum_loss = jnp.inf
            best_model_shard_id = self.params["current"]
            # self.time_interval["current"] = jnp.array([shard_id * self.time_per_shard, (shard_id+1) * self.time_per_shard])
            self.time_interval["current"] = jnp.array([0, self.time_per_shard])
            # initialize the opt_state
            opt_state = self.optimizer.init(self.params["current"])
            rngs = jax.random.split(rng_shard, self.cfg.train.number_of_iterations)
            for epoch in range(self.cfg.train.number_of_iterations):
                rng = rngs[epoch]
                rng_train, rng_test, rng_plot = random.split(rng, 3)

                v_g_etc = value_and_grad_fn(self.params, self.time_interval, rng_train)
                self.params["current"], opt_state = step_fn(self.params["current"], opt_state, v_g_etc["grad"])
                if v_g_etc["loss"] < minimum_loss:
                    best_model_shard_id = self.params["current"]
                    minimum_loss = v_g_etc["loss"]

                v_g_etc.pop("grad")
                params_norm = compute_pytree_norm(self.params["current"])
                v_g_etc["params_norm"] = params_norm

                wandb.log({f"shard {shard_id}": v_g_etc}, step=epoch + shard_id * self.cfg.train.number_of_iterations)
                if self.cfg.pde_instance.perform_test and (epoch % self.cfg.test.frequency == 0 or epoch >= self.cfg.train.number_of_iterations - 3):
                    result_epoch = test_fn(self.params, self.time_interval, rng_test)
                    wandb.log({f"shard {shard_id}": result_epoch}, step=epoch + shard_id * self.cfg.train.number_of_iterations)
                    if self.cfg.test.verbose:
                        msg = f"In epoch {epoch + 1: 5d}, "
                        for key in v_g_etc:
                            msg = msg + f"{key} is {v_g_etc[key]: .3e}, "
                        for key in result_epoch:
                            msg = msg + f"{key} is {result_epoch[key]: .3e}, "
                        print(msg)
                # if (epoch + 1) % self.cfg.plot.frequency == 0 and self.method.plot_fn is not None: plot(self.params, rng_plot)

            minimum_loss_collection.append(minimum_loss)
            # store the params for a specific time shard
            self.params["previous"].append(copy.deepcopy(best_model_shard_id))
            self.time_interval["previous"].append(copy.deepcopy(self.time_interval["current"]))
            self.params["current"] = copy.deepcopy(best_model_shard_id)

            plot_fn(self.params, self.time_interval, rng_plot) 

            # TODO: pretrain if necessary
            # TODO: save model
            # TODO: adapt other equations
            # TODO: evaluate the metric, e.g. trend to equilibrium, flocking, Landau damping

