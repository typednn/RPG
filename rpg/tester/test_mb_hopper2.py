import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer

#env = GymVecEnv('Humanoid-v3', 1, ignore_truncated_done=True)
env = GymVecEnv('Hopper-v3', 1, ignore_truncated_done=True)

trainer = Trainer.parse(
    env, buffer=dict(), head=dict(std_mode='statewise', std_scale=1., squash=True), enta=dict(coef=1.),  optim=dict(max_grad_norm=1., lr=0.0001), update_train_step=1, have_done=False, hooks=dict(evaluate_pi=dict()), horizon=1, weights=dict(done=1000.),
    worldmodel=dict(detach_hidden=False),

    qmode='Q',

    wandb=dict(name='hopper2-q'),
    _variants=dict(
        ln=dict(state_layer_norm=True, path='tmp/hopper_ln', weights=dict(state=10000.)),
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
