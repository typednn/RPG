# https://www.notion.so/Exp-list-1-18-3e5496395d9c4af6bf1a48dc360bc33b


# kitchen
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4

# antfall
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0 
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 2 

python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4 


# cabinet
# sac
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 5

python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 5
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 6


# stick pull
python3 maze_tester/exp_sparse.py --exp stickpull --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 0
##  gaussian 002
python3 maze_tester/exp_sparse.py --exp stickpull --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 1
# redo .. 002
python3 maze_tester/exp_sparse.py --exp stickpull --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 2

# block
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 1

python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 11
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 11



# dense ant

python3 maze_tester/exp_dense.py --exp denseant --seed 5,6 --runall remote_parallel --wandb True --ids 0 --silent --cpu 5
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 7


# dense cabinet, exp increase

python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 9
python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 10


python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 7
python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 7