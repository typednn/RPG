import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg
from nn.space import Discrete


# N = 1
N = 256
env = TorchEnv('SmallMaze', N, reward=False)

hidden_space  = Discrete(6)

train_rpg.parse(
    env, 
    hidden_space,

    steps=40, obs_norm=False, reward_norm=True,

    hidden_head=dict(epsilon=0.),

    actor=dict(
        head=dict(linear=False, std_scale=0.6, std_mode='statewise')
    ),
    relbo=dict(prior=0.01, mutual_info=1., ent_a=1., ent_z=1.),

    gae=dict(lmbda=0.97),
    batch_size=2000,
    ppo=dict(learning_epoch=5),

    hooks=dict(
        #save_model=dict(n_epoch=10),
        plot_maze_env_rnd=dict(resolution=64),
        log_info=dict(n_epoch=1),
        save_traj=dict(n_epoch=1, save_gif_epochs=10),
    ),
    rnd=dict(learning_epoch=2), # no rnd
    path='tmp/rnd/rpg',
) # use tanh