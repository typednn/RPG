python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 8

python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 8


python3 maze_tester/exp_dense.py --exp densefall --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 0
python3 maze_tester/exp_dense.py --exp densefall --runall remote_parallel  --wandb True --cpu 5 --seed 0,1 --silent --ids 8


python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  6

python3 maze_tester/exp_sparse.py --exp stickpull --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids  11