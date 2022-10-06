import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

# N = 1
N = 10
env = GymVecEnv('Swimmer-v2', N, ignore_truncated_done=False)

# swimmer must clip the done in the end.
train_ppo.parse(env, steps=2000, obs_norm=True, actor=dict(head=dict(linear=False, std_scale=0.6, std_mode='fix_no_grad')), gae=dict(correct_gae=True, gamma=0.999), batch_size=2000, ppo=dict(learning_epoch=10)) # use tanh