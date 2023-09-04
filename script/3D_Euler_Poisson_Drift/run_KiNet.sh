args=(--PDE 3D-Euler-Poisson-Drift
      --method KiNet
      --boundary_condition None
      --domain_dim 3
      --number_of_iterations 40000
      --learning_rate 1e-3
      --total_evolving_time 2
      --batch_size_initial 100
      --batch_size_ref 50000
      --batch_size_test_ref 60000
      --ODE_tolerance 1e-6
      --test_frequency 50
      --plot_frequency 99999
)

CUDA_VISIBLE_DEVICES=1 python main.py "${args[@]}"