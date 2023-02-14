# reward schedule
# https://www.notion.so/Exp-lists-1-17-851825742a3c4f04a2c835701d9d2cb7

# dense 
#python3 maze_tester/exp_dense.py --exp densecabinet --seed 1,2 --runall parallel --wandb False --seed 1,2 --id 7
#python3 maze_tester/exp_dense.py --exp denseant --seed 1,2 --runall parallel --wandb False --seed 1,2 --id 7
#python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4,5
#python3 maze_tester/exp_dense.py --exp densecabinet --seed 1,2 --runall parallel --wandb False --seed 1,2 --id 9
# python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 8


# dense ant baseline
python3 maze_tester/exp_dense.py --exp denseant --seed 1,2 --runall remote_parallel --wandb True --ids 0 --silent
python3 maze_tester/exp_dense.py --exp denseant --seed 3,4 --runall remote_parallel --wandb False --ids 0

# dense ant, increasing reward
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 9

# cabinet sac baseline
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0


# block sac baseline
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0

## search parameters for block
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 1,2


## debug for kitchen
python3 maze_tester/exp_sparse.py --exp kitchen --wandb False --cpu 5  --id 0 --opt.hooks.save_train_occupancy.n_epoch 1


# antpush baseline
python3 maze_tester/exp_dense.py --exp denseant --seed 1,2 --runall parallel --wandb False --id 0
python3 maze_tester/exp_dense.py --exp denseant --seed 3,4 --runall parallel --wandb False --id 0

# antfall
python3 maze_tester/exp_sparse.py --exp fall --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0 


# kitchen sac baseline
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp block --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0