import optax
import jax
import jax.numpy as jnp
from tqdm import trange
from utils.logging_utils import save_to_csv
from api import Method
import jax.random as random
from optax import GradientTransformation

import sys


class JaxTrainer:
    def __init__(self,
                 args,
                 method: Method,
                 rng: jnp.ndarray,
                 save_directory: str,
                 optimizer: GradientTransformation,
                 forward_fn,
                 params: optax.Params,
                 ):
        self.args = args
        self.forward_fn = forward_fn
        self.params = params
        self.optimizer = optimizer
        self.method = method
        self.rng = rng
        self.save_directory = save_directory

    def fit(self, ):

        # initialize the opt_state
        opt_state = self.optimizer.init(self.params)

        # jit or pmap the gradient computation for efficiency
        def value_and_grad_fn(params, rng):
            return self.method.value_and_grad_fn(self.forward_fn, params, rng)


        if self.args.use_pmap_train and jax.local_device_count() > 1:
            _value_and_grad_fn = jax.pmap(value_and_grad_fn, in_axes=(None, 0))

            def value_and_grad_fn_efficient(params, rng):

                rngs = random.split(rng, jax.local_device_count())
                # compute in parallel
                v_g_etc = _value_and_grad_fn(params, rngs)
                v_g_etc = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), v_g_etc)
                return v_g_etc
                # loss_values, grads = _value_and_grad_fn(params, rngs)
                #
                # # aggregate the result
                # grad = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), grads)
                # loss_value = jnp.mean(loss_values)
                #
                # return loss_value, grad
        else:
            value_and_grad_fn_efficient = jax.jit(value_and_grad_fn)
            # value_and_grad_fn_efficient = value_and_grad_fn

        @jax.jit
        def step(params, opt_state, grad):
            updates, opt_state = self.optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        @jax.jit
        def test(params, rng):
            return self.method.test_fn(self.forward_fn, params, rng)

        # @jax.jit
        def plot(params, rng):
            return self.method.plot_fn(self.forward_fn, params, rng)

        save_file = f"{self.save_directory}/results.csv"
        epoch_list = []
        objetive_value_list = []
        results = {}
        rngs = jax.random.split(self.rng, self.args.number_of_iterations)
        for epoch in range(self.args.number_of_iterations):
            rng = rngs[epoch]
            rng_train, rng_test, rng_plot = random.split(rng, 3)

            v_g_etc = value_and_grad_fn_efficient(self.params, rng_train)
            self.params, opt_state = step(self.params, opt_state, v_g_etc["grad"])

            if (epoch % self.args.test_frequency == 0 and self.method.test_fn is not None) or epoch >= self.args.number_of_iterations - 3:
                epoch_list.append(epoch + 1)
                objetive_value_list.append(v_g_etc["loss"])
                result_epoch = test(self.params, rng_test)
                if len(results) == 0:
                    for key in result_epoch:
                        results[key] = [result_epoch[key]]
                else:
                    for key in result_epoch:
                        results[key].append(result_epoch[key])
                # msg = f"In epoch {epoch + 1: 5d}, loss is {loss_value: .3e}, "
                msg = f"In epoch {epoch + 1: 5d}, "
                for key in v_g_etc:
                    if key != "grad":
                        msg = msg + f"{key} is {v_g_etc[key]: .3e}, "
                for key in result_epoch:
                    msg = msg + f"{key} is {result_epoch[key]: .3e}, "
                print(msg); sys.stdout.flush()

            if (epoch + 1) % self.args.plot_frequency == 0 and self.method.plot_fn is not None: plot(self.params, rng_plot)

            if (epoch + 1) % self.args.save_frequency == 0:
                results['steps'] = epoch_list
                results['objective value'] = objetive_value_list
                save_to_csv(results, save_file)

        results['steps'] = epoch_list
        results['objective value'] = objetive_value_list
        save_to_csv(results, save_file)

        return self.params
