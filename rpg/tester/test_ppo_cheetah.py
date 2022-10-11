import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

# N = 1
N = 10
env = GymVecEnv('HalfCheetah-v3', N)

train_ppo.parse(
    env, steps=2000, obs_norm=True, actor=dict(head=dict(linear=True, std_scale=0.5, std_mode='fix_learnable')), 
    batch_size=2000, ppo=dict(learning_epoch=2, ignore_done=True),

    hooks=dict(
        save_model=dict(n_epoch=10), log_info=dict(n_epoch=1),
        monitor_action_std=dict(n_epoch=1), 
            #std_decay=dict(TYPE='MultiStepScheduler', # milestones=[1e7, 1e7, 3e7, 4e7, 5e7, 6e7,7e7], gamma=0.6)),
    )
) # use tanh