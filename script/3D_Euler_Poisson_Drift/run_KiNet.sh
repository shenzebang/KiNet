args=(--PDE 3D-Euler-Poisson-Drift
      --method KiNet
      --boundary_condition None
      --domain_dim 3
      --number_of_iterations 80000
      --learning_rate 1e-3
      --total_evolving_time 3
      --batch_size_initial 64
      --batch_size_ref 50000
      --batch_size_test_ref 60000
      --ODE_tolerance 1e-5
      --test_frequency 100
      --plot_frequency 99999
#      --use_pmap_train
)

CUDA_VISIBLE_DEVICES=3 python main.py "${args[@]}"