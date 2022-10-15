import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer


# N = 1
#N = 1
env = GymVecEnv('HalfCheetah-v3', 10, ignore_truncated_done=True)
# env = TorchEnv('SmallMaze', n=100, ignore_truncated=True, reward=True)


trainer = Trainer(env)
trainer.run_rpgm()