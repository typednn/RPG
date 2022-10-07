import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.ppo import train_ppo

# N = 1
N = 256
env = TorchEnv('SmallMaze', N)

train_ppo.parse(
    env, steps=40, obs_norm=False, reward_norm=True,

    actor=dict(
        head=dict(linear=False, std_scale=0.6, std_mode='statewise')
    ),

    gae=dict(correct_gae=True, ignore_done=False, lmbda=0.97),
    batch_size=2000,
    ppo=dict(learning_epoch=5),

    hooks=dict(
        #save_model=dict(n_epoch=10),
        plot_maze_env_rnd=dict(resolution=64),
        log_info=dict(n_epoch=1),
        save_traj=dict(n_epoch=1, save_gif_epochs=10),
    ),
    rnd=dict(learning_epoch=2), # no rnd
    path='tmp/rnd/ppo',
) # use tanh