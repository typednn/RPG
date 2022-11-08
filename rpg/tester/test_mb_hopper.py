import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer

#env = GymVecEnv('Humanoid-v3', 1, ignore_truncated_done=True)
env = GymVecEnv('Hopper-v3', 1, ignore_truncated_done=True)

trainer = Trainer.parse(
    env, buffer=dict(), head=dict(std_mode='statewise', std_scale=1., squash=True), enta=dict(coef=1.),  optim=dict(max_grad_norm=1., lr=0.0001), update_train_step=1, have_done=True, hooks=dict(evaluate_pi=dict()), horizon=3, weights=dict(done=100.),

    wandb=dict(name='hopper'),
    _variants=dict(
        ln=dict(state_layer_norm=True, path='tmp/hopper_ln', weights=dict(state=20000.)),
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
