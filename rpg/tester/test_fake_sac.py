import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
#from rpg.rpgm import Trainer
from rpg.sac import SAC


# N = 1
#N = 1
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = SAC.parse(env, buffer=dict(priority=False), head=dict(std_scale=1.), qnet=dict(predict_q=True, gamma=0.99), action_penalty=0.0, entropy_target=-6)
# env = TorchEnv('SmallMaze', n=100, ignore_truncated=True, reward=True)
# trainer = SAC.parse(env, update_step=10, buffer=dict(priority=False), qnet=dict(predict_q=False), hooks=dict(save_traj=dict(n_epoch=2, save_gif_epochs=10)), action_penalty=0.0, entropy_target=-2)
trainer.run_rpgm()