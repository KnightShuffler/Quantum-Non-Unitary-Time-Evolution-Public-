nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 2 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 01 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n2d2.log --gpu_solve --gpu_sim &

nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n4d2-none.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift a --runs 25 --run_log lr_n4d2-a.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift theta_2pi --runs 25 --run_log lr_n4d2-theta_2pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift theta_pi_pi --runs 25 --run_log lr_n4d2-theta_pi_pi.log --gpu_solve --gpu_sim &

nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n4d4-none.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift a --runs 25 --run_log lr_n4d4-a.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift theta_2pi --runs 25 --run_log lr_n4d4-theta_2pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 4 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 0101 --log_path ./qite_logs/server_runs --drift theta_pi_pi --runs 25 --run_log lr_n4d4-theta_pi_pi.log --gpu_solve --gpu_sim &

nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n6d2-none.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift a --runs 10 --run_log lr_n6d2-a.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_2pi --runs 10 --run_log lr_n6d2-theta_2pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 2 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_pi_pi --runs 10 --run_log lr_n6d2-theta_pi_pi.log --gpu_solve --gpu_sim &

nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift a --runs 10 --run_log lr_n6d4-a.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_2pi --runs 10 --run_log lr_n6d4-theta_2pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_pi_pi --runs 10 --run_log lr_n6d4-theta_pi_pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 4 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n6d4-none.log --gpu_solve --gpu_sim &

nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 6 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift none --runs 1 --run_log lr_n6d6-none.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 6 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift a --runs 10 --run_log lr_n6d6-a.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 6 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_2pi --runs 10 --run_log lr_n6d6-theta_2pi.log --gpu_solve --gpu_sim &
nohup python qite_cli.py -db 0.1 -delta 0.1 -N 30 -n 6 -D 6 -H lr_heisenberg -J 1 1 1 --init_label 010101 --log_path ./qite_logs/server_runs --drift theta_pi_pi --runs 10 --run_log lr_n6d6-theta_pi_pi.log --gpu_solve --gpu_sim &