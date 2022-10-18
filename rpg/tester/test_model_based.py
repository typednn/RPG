import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer


# N = 1
#N = 1
# env = TorchEnv('SmallMaze', n=100, ignore_truncated=True, reward=True)

# trainer = Trainer.parse(env, update_step=10, buffer=dict(priority=False), head=dict(squash=True), hooks=dict(save_traj=dict(n_epoch=2, save_gif_epochs=10)))
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = Trainer.parse(env, max_update_step=1000, buffer=dict(priority=False), head=dict(std_mode='statewise', std_scale=1., squash=True), entropy_coef=1., entropy_target=-6, actor_optim=dict(max_grad_norm=1.)) # do not know if we need max_grad_norm
# trainer = Trainer.parse(env, update_step=200, buffer=dict(priority=False), head=dict(std_mode='statewise', std_scale=1., squash=True), entropy_coef=1., entropy_target=-6, actor_optim=dict(max_grad_norm=1.), weights=dict(state=100., prefix=0.5, value=0.5)) # do not know if we need max_grad_norm
# trainer = Trainer.parse(env, update_step=1000, buffer=dict(priority=False), head=dict(std_mode='statewise', std_scale=1., squash=True), entropy_coef=1., entropy_target=-6, actor_optim=dict(max_grad_norm=1.), critic_weight=1.) # do not know if we need max_grad_norm
trainer.run_rpgm()
