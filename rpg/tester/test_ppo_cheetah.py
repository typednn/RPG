import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo

N = 1
env = GymVecEnv('HalfCheetah-v3', N)

train_ppo(env)