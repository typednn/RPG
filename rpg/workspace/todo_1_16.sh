# Kitchen SAC baseline, 4 seed; 2 process
# Waiting for the result of kichen; new version
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_sparse.py --exp kitchen --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0

# AntPush SAC baseline, 2 seed; 1 prcess;
python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 3,4 --silent --ids 0


# AntPush test RPGD, 2 seed; search for coef ..;  3 process
# discrete001, 005, 0005
python3 maze_tester/exp_sparse.py --exp ant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 5,6,7

# Cabinet, test RPGD;   3 process
python3 maze_tester/exp_sparse.py --exp cabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 5,6,7


# AntPushDense, test RPGD;
# ant sac baseline; 4 process
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0

#python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 1,2,3
python3 maze_tester/exp_dense.py --exp denseant --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 3,4,5


# CabinetDense, sac baselines
#python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 0
python3 maze_tester/exp_dense.py --exp densecabinet --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent --ids 4,5

