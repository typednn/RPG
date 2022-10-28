import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.ppo import train_ppo

N = 100
env = TorchEnv('TripleMove', N, ignore_truncated_done=True, n_goals=1)
train_ppo.parse(env, obs_norm=False, steps=env.max_time_steps, hooks=dict(save_traj=dict(n_epoch=1)), path='tmp/oc')