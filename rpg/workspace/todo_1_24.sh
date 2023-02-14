# dense cabinet sac
# python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0,1
# python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0,1
# python3 maze_tester/exp_sparse.py --exp door --runall remote_parallel  --wandb True --cpu 5 --seed 4,5 --silent --ids 0,1

# python3 maze_tester/exp_sparse.py --exp kitchen2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 1
# python3 maze_tester/exp_sparse.py --exp kitchen2 --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 1

# python3 maze_tester/exp_sparse.py --exp kitchen2 --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4
# python3 maze_tester/exp_sparse.py --exp kitchen2 --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 4

#python3 maze_tester/exp_sparse.py --exp kitchen2 --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 0,1,4

#python3 maze_tester/exp_sparse.py --exp kitchen4 --runall remote  --wandb True --cpu 5 --seed 1,2,3,4,5 --silent --ids 4
#python3 maze_tester/exp_sparse.py --exp kitchen4 --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 1
#python3 maze_tester/exp_sparse.py --exp kitchen4 --runall remote_parallel  --wandb True --cpu 5 --seed 5,6 --silent --ids 1

# # sac again
# python3 maze_tester/exp_denseant.py --exp denseantabl --seed 5 --runall remote --ids 0 --wandb True --cpu 3 --silent

# # # seg3n1gamma for safety
# python3 maze_tester/exp_denseant.py --exp denseantabl --seed 1,2,3,4,5 --runall remote --ids 6,8 --wandb True --silent

# # seg3n1gamma for safety
# python3 maze_tester/exp_denseant.py --exp denseantabl --seed 1,2 --runall remote --ids 9 --wandb True --silent
python3 maze_tester/exp_draw.py --exp drawblock --seed 1,2 --runall remote --ids 1 --wandb True --silent
python3 maze_tester/exp_draw.py --exp drawstickpull --seed 1,2 --runall remote --ids 1 --wandb True --silent
python3 maze_tester/exp_draw.py --exp drawcabinet --seed 1,2 --runall remote --ids 1 --wandb True --silent