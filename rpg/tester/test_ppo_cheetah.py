import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

# N = 1
N = 10
env = GymVecEnv('HalfCheetah-v3', N)

train_ppo.parse(env, steps=2000, obs_norm=True, actor=dict(head=dict(linear=False, std_scale=0.36, std_mode='statewise')), gae=dict(correct_gae=True), batch_size=2000, ppo=dict(learning_epoch=2)) # use tanh