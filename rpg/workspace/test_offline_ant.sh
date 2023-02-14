# create new ant env reward 

# the code to create the buffer ..
# python3 maze_tester/test_maze.py --var ant_squash  --reward_scale 0. --path tmp/ant2_buffer --env_name AntMaze2 --max_epoch 8001 --save_buffer_epoch 1000

# the code to run with relabeled reward
#python3 maze_tester/test_maze.py --var ant_squash --fix_buffer tmp/ant2_buffer/buffer.pt --max_epoch 2000 --path tmp/ant2_relabel2 --env_name AntMaze2 --rnd.scale 0. --reward_relabel ant2 --hidden.n 1


python3 maze_tester/test_maze.py --var ant_squash --fix_buffer tmp/ant2_buffer/buffer.pt --max_epoch 10000 --path tmp/ant2_relabel3 --env_name AntMaze2 --rnd.scale 0. --reward_relabel ant2 --hidden.n 1 --env_cfg.reset_loc True --hooks.accumulate_ant_traj.n_epoch 10