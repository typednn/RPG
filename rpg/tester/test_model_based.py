import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer

# max_grad_norm=1.
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = Trainer.parse(env, max_update_step=0, head=dict(std_mode='statewise', std_scale=1., squash=True), entropy_coef=1., entropy_target=-6, actor_optim=dict(max_grad_norm=1.), horizon=3, update_train_step=1, hooks=dict(evaluate_pi=dict())) # do not know if we need max_grad_norm
trainer.run_rpgm()
