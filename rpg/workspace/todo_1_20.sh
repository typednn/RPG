# dense cabinet sac
python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 0
python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 2,3 --silent --ids 0


python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 0,1 --silent --ids 9
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 2,3 --silent --ids 9

python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 0,1 --silent --ids 10



python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  11
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  12
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids  12


# blockpush
python3 maze_tester/exp_sparse.py --exp block3 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  11
python3 maze_tester/exp_sparse.py --exp block3 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  3

python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  11
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  11

python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  12
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  12

python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  8
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  8

python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  9
python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids  9




python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids  0
python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 2,3 --silent --ids  0

python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids  1
python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 2,3 --silent --ids  1

python3 maze_tester/exp_dense.py --exp densefall --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 7

python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 0,1 --silent --ids 8
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 2 --seed 2,3 --silent --ids 8


python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids  0
python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 2,3 --silent --ids  0

python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids  1
python3 maze_tester/exp_sparse.py --exp ball --runall remote_parallel  --wandb True --cpu 5 --seed 2,3 --silent --ids  1