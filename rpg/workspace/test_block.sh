# what we want to have: pd

#python3 maze_tester/test_maze.py --var block  --reward_scale 0. --hooks.save_traj.n_epoch 20 --save_video 300 --env_cfg.n_block 1 #--eval_episode 1
#python3 maze_tester/test_maze.py --path tmp/blockent --var block  --reward_scale 0. --hooks.save_traj.n_epoch 20 --save_video 300 --env_cfg.n_block 1 --pi_a.ent.coef 0.01 #--eval_episode 1
#python3 maze_tester/test_maze.py --path tmp/block3 --var block  --reward_scale 0. --hooks.save_traj.n_epoch 20 --save_video 300 --env_cfg.n_block 2 #--eval_episode 1

python3 maze_tester/exp_1_4.py --exp blocktry --runall remote --wandb True  --silent --seed 1