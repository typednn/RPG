import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer

# max_grad_norm=1.
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = Trainer.parse(
    env, 
    # head=dict(std_mode='statewise', std_scale=1., squash=True), 
    # enta=dict(coef=1., target=-6), 
    # optim=dict(max_grad_norm=1.),
    update_train_step=1,
    horizon=3,
    hooks=dict(evaluate_pi=dict()),

    # wandb=dict(),
    _variants=dict(
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
