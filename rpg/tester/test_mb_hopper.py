import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer

#env = GymVecEnv('Humanoid-v3', 1, ignore_truncated_done=True)
env = GymVecEnv('Hopper-v3', 1, ignore_truncated_done=True)

trainer = Trainer.parse(env, max_update_step=0, buffer=dict(), head=dict(std_mode='statewise', std_scale=1., squash=True), entropy_coef=1., entropy_target=100, actor_optim=dict(max_grad_norm=1.), dyna_optim=dict(max_grad_norm=1.), update_train_step=1, have_done=True, hooks=dict(evaluate_pi=dict()), horizon=3, zero_done_value=False, qnet=dict(gamma=0.99), pg=False, done_penalty=10., weights=dict(done=100.)) # do not know if we need max_grad_norm
trainer.run_rpgm()
