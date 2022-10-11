import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

# N = 1
N = 10
env = GymVecEnv('Swimmer-v2', N, ignore_truncated_done=False)

# swimmer must clip the done in the end.
train_ppo.parse(env, steps=2000, obs_norm=True, actor=dict(head=dict(linear=False, std_scale=0.5, std_mode='fix_no_grad')), 
                batch_size=2000, ppo=dict(gamma=0.999, learning_epoch=10),
                hooks=dict(log_info=dict(n_epoch=10), monitor_action_std=dict(n_epoch=1, std_decay=dict(TYPE='MultiStepScheduler', milestones=[1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 8e6,10e6], gamma=0.8))))