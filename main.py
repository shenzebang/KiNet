import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import optax
import jax.random as random
from core.trainer import JaxTrainer
from register import get_pde_instance, get_method


def get_optimizer(train_cfg: DictConfig):
    optimizer_cfg = train_cfg.optimizer
    if optimizer_cfg.learning_rate.scheduling == "None":
            lr_schedule = optimizer_cfg.learning_rate.initial
    elif optimizer_cfg.learning_rate.scheduling == "cosine":
        lr_schedule = optax.cosine_decay_schedule(optimizer_cfg.learning_rate.initial, train_cfg.number_of_iterations, 0.1)
    elif optimizer_cfg.learning_rate.scheduling == "warmup-cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=optimizer_cfg.learning_rate.initial,peak_value=optimizer_cfg.learning_rate.initial,
                                                         warmup_steps=train_cfg.number_of_iterations // 4, decay_steps=train_cfg.number_of_iterations, 
                                                         end_value=optimizer_cfg.learning_rate.initial * .1)
    else:
        raise NotImplementedError
    if optimizer_cfg.grad_clipping.type=="adaptive":
        clip = optax.adaptive_grad_clip
    elif optimizer_cfg.grad_clipping.type=="non-adaptive":
        clip = optax.clip
    elif optimizer_cfg.grad_clipping.type=="global":
        clip = optax.clip_by_global_norm
    else:
        raise ValueError("type of clipping should be either adaptive or non-adaptive!")
    if optimizer_cfg.method == "SGD":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.sgd(learning_rate=lr_schedule, momentum=optimizer_cfg.momentum)
                                )
    elif optimizer_cfg.method == "ADAM":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.adam(learning_rate=lr_schedule, b1=optimizer_cfg.momentum)
                                )
    elif optimizer_cfg.method == "ADAGRAD":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.adagrad(learning_rate=lr_schedule)
                                )
    else:
        raise NotImplementedError
    return optimizer


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    wandb.login()
    pde_instance_name = f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"
    run = wandb.init(
        # Set the project where this run will be logged
        project=f"{pde_instance_name}-{cfg.solver.name}-{cfg.pde_instance.total_evolving_time}",
        # Track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg),
        # hyperparameter tuning mode or normal mode.
        # name=cfg.mode
    )

    rng_problem, rng_method, rng_trainer = random.split(random.PRNGKey(cfg.seed), 3)

    # create problem instance
    pde_instance = get_pde_instance(cfg)(cfg=cfg, rng=rng_problem)

    # create method instance
    method = get_method(cfg)(pde_instance=pde_instance, cfg=cfg, rng=rng_method)

    # create model
    net, params = method.create_model_fn()

    # create optimizer
    optimizer = get_optimizer(cfg.train)

    # Construct the JaxTrainer
    trainer = JaxTrainer(cfg=cfg, method=method, rng=rng_trainer, forward_fn=net.apply,
                         params=params, optimizer=optimizer)

    # Fit the model
    params_trained = trainer.fit()

    # Test the model

    wandb.finish()


if __name__ == '__main__':
    main()
