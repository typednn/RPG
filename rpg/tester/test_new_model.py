import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer

# max_grad_norm=1.
env = GymVecEnv('HalfCheetah-v3', 5, ignore_truncated_done=True)
# env = GymVecEnv('PixelCheetah', 1, ignore_truncated_done=True)
trainer = Trainer.parse(
    env, 
    model=dict(qmode='value'),
    update_train_step=1,
    horizon=3,
    hooks=dict(evaluate_pi=dict()),
    batch_size=256,
    buffer=dict(max_episode_num=200),
    _variants=dict(
    ),
    cem = dict(seed_steps=1),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
