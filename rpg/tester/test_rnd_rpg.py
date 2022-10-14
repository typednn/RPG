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

    steps=40, obs_norm=False,

    hidden_head=dict(epsilon=0.),

    actor=dict(
        head=dict(linear=False, std_scale=0.3, std_mode='fix_no_grad')
    ),
    relbo=dict(prior=0.0, mutual_info=0.1),

    # gae=dict(lmbda=0.97),
    batch_size=2000,
    ppo=dict(learning_epoch=2, entropy=dict(coef=0.01, target=None)),
    ppo_higher=dict(learning_epoch=2, entropy=dict(coef=1., target=None)),

    hooks=dict(
        #save_model=dict(n_epoch=10),
        plot_maze_env_rnd=dict(resolution=64),
        log_info=dict(n_epoch=1),
        save_traj=dict(n_epoch=2, save_gif_epochs=10),
    ),
    rnd=dict(learning_epoch=10, rnd_scale=0.1), # no rnd
    path='tmp/rnd/rpg2',
) # use tanh