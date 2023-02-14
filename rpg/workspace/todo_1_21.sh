# dense cabinet sac
python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0

python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 8
python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 9

python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 3
python3 maze_tester/exp_sparse.py --exp ant2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4


python3 maze_tester/exp_dense_v2.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 0
python3 maze_tester/exp_dense_v2.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 1
python3 maze_tester/exp_dense_v2.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 2
python3 maze_tester/exp_dense_v2.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 3
python3 maze_tester/exp_dense_v2.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 4


python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  0
python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids  1


python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 0,1 --silent --ids 9
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 2,3 --silent --ids 10