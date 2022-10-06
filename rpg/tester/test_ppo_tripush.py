import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.ppo import train_ppo

N = 100
env = TorchEnv('TripleMove', N)

train_ppo.parse(env, steps=env.max_time_steps, hooks=dict(save_traj=dict(n_epoch=1)))