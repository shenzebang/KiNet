CUDA_VISIBLE_DEVICES=0 \
python main.py \
    pde_instance=nd_fokker_planck \
    pde_instance.total_evolving_time=10 \
    pde_instance.domain_dim=10 \
    train.optimizer.weight_decay=0.0005 \
    solver.train.batch_size_ref=0 \
    neural_network.hidden_dim=64 \
    neural_network.layers=3