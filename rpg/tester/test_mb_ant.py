import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer

env = GymVecEnv('Ant-v3', 1, ignore_truncated_done=True)
trainer = Trainer.parse(env, head=dict(std_mode='statewise', std_scale=1., squash=True), optim=dict(max_grad_norm=1., lr=0.0001), hooks=dict(evaluate_pi=dict()), have_done=True, horizon=3, weights=dict(done=100.)) # do not know if we need max_grad_norm
trainer.run_rpgm()
