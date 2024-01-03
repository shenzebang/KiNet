CUDA_VISIBLE_DEVICES=1 \
    python main.py \
        pde_instance.total_evolving_time=2.0 \
        train.optimizer.weight_decay=0 \
        train.batch_size=256 \
        neural_network.hidden_dim=128 \
        neural_network.layers=3 \
        neural_network.activation=relu \
        plot.frequency=999999 \
        train.optimizer.learning_rate.initial=1e-3 \
        train.optimizer.method=ADAM \
        train.optimizer.grad_clipping.threshold=999999 \
        train.optimizer.learning_rate.scheduling=warmup-cosine \
        train.number_of_time_shard=10 \
        train.number_of_iterations=80000