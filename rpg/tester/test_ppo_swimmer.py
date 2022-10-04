import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

# N = 1
N = 10
env = GymVecEnv('Swimmer-v2', N)

# swimmer must clip the done in the end.
train_ppo.parse(env, steps=2000, obs_norm=True, actor=dict(head=dict(linear=False)), gae=dict(correct_gae=False, gamma=0.999, use_env_done=False), batch_size=2000) # use tanh