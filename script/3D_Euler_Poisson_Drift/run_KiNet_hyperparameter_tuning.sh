CUDA_VISIBLE_DEVICES=3 python main.py --multirun train.optimizer.weight_decay=0.001,0.0005,0.00025 neural_network.hidden_dim=40,64,128 neural_network.layers=2,3 plot.frequency=999999