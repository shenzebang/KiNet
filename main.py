import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from config import args_parser
import optax
import jax.random as random
import jax.numpy as jnp
from core.trainer import JaxTrainer
import jax
from utils.logging_utils import save_config
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from methods.KiNet import KiNet
from methods.PINN import PINN
# Example problems
PDE_INSTANCES = {
    '2D-Kinetic-Fokker-Planck': KineticFokkerPlanck,
}
# Methods
METHODS = {
    'KiNet' : KiNet,
    'PINN'  : PINN,
}


if __name__ == '__main__':
    args = args_parser()

    save_directory = f"./{args.save_directory}/{args.PDE}_{args.method}_{args.total_evolving_time}"
    save_config(save_directory, args)
    print(f"[Running {args.method} on {args.PDE} with a total evolving time of {args.total_evolving_time}]")

    rng_problem, rng_method, rng_trainer = random.split(random.PRNGKey(args.seed), 3)


    # create problem instance
    pde_instance = PDE_INSTANCES[args.PDE](args=args, rng=rng_problem)

    # create method instance
    method = METHODS[args.method](pde_instance=pde_instance, args=args, rng=rng_method)

    # create model
    # net = method.create_model_fn()
    # params = net.init(random.PRNGKey(11), jnp.zeros(1), jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))))
    net, params = method.create_model_fn()

    # create optimizer
    optimizer = optax.adam(learning_rate=args.learning_rate)

    # Construct the JaxTrainer
    trainer = JaxTrainer(args=args, method=method, rng=rng_trainer, save_directory=save_directory, forward_fn=net.apply, params=params, optimizer=optimizer)

    # Fit the model
    params_trained = trainer.fit()

    # Test the model

