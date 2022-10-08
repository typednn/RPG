import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg
from nn.space import Discrete


# N = 1
N = 256
env = TorchEnv('SmallMaze', N, reward=False)

hidden_space  = Discrete(4)

train_rpg.parse(
    env, 
    hidden_space,

    steps=40, obs_norm=False, reward_norm=True,

    hidden_head=dict(epsilon=0.),

    actor=dict(
        head=dict(linear=False, std_scale=0.2, std_mode='fix_no_grad')
    ),
    relbo=dict(prior=0.0, mutual_info=0.01, ent_a=0.0, ent_z=0.),

    gae=dict(lmbda=0.97),
    batch_size=2000,
    ppo=dict(learning_epoch=5),
    ppo_higher=dict(learning_epoch=0),

    hooks=dict(
        #save_model=dict(n_epoch=10),
        plot_maze_env_rnd=dict(resolution=64),
        log_info=dict(n_epoch=1),
        save_traj=dict(n_epoch=1, save_gif_epochs=10),
    ),
    rnd=dict(learning_epoch=2), # no rnd
    path='tmp/rnd/rpg_no_high_level',
) # use tanh