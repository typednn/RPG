import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer
from rpg.oc import OptionCritic

# max_grad_norm=1.
N = 1
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=1)

trainer = Trainer.parse(
    env, max_update_step=0,
    buffer=dict(max_episode_num=20000),
    head=dict(
        std_mode='statewise',
        std_scale=1.,
        squash=True
    ),
    entropy_coef=0.0001,
    entropy_target=-2.,
    actor_optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=5,
    actor_delay=2, #10,
    # tau = 0.001,
    #tau=0.001,
    update_train_step=1,
    hooks=dict(save_traj=dict(n_epoch=10)),
    path='tmp/oc',
    weights=dict(prefix=1.),
    pg=True
) # do not know if we need max_grad_norm
trainer.run_rpgm()
