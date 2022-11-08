import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer

# max_grad_norm=1.
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = Trainer.parse(
    env, 
    head=dict(std_mode='statewise', std_scale=1., squash=True), 
    enta=dict(coef=1., target=-6), 
    optim=dict(max_grad_norm=1.),
    horizon=3,
    update_train_step=1,
    hooks=dict(evaluate_pi=dict()),

    wandb=dict(name='cheetah'),
    _variants=dict(
        ln=dict(state_layer_norm=True, path='tmp/cheetah_ln', weights=dict(state=10000.)),
        ln2=dict(state_layer_norm=True, path='tmp/cheetah_ln', weights=dict(state=100.)),
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
