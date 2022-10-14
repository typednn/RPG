import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer


# N = 1
N = 1
env = GymVecEnv('HalfCheetah-v3', N, ignore_truncated_done=True)


trainer = Trainer(env)
trainer.run_rpgm()