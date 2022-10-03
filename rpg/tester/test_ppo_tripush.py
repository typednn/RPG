import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.ppo import train_ppo

N = 100
env = TorchEnv('TripleMove', N)

train_ppo(env, steps=env.max_time_steps)