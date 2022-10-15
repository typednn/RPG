import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
#from rpg.rpgm import Trainer
from rpg.sac import SAC


# N = 1
#N = 1
#env = GymVecEnv('HalfCheetah-v3', 10, ignore_truncated_done=True)
env = TorchEnv('SmallMaze', n=100, ignore_truncated=True, reward=True)


trainer = SAC(env, update_step=100, buffer=dict(priority=False), qnet=dict(predict_q=True))
trainer.run_rpgm()