import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer

# max_grad_norm=1.
# env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
if __name__ == '__main__':
    env = GymVecEnv('PixelCheetah', 1, ignore_truncated_done=True)
    trainer = Trainer.parse(
        env, 
        model=dict(qmode='value'),
        #update_train_step=1,
        update_train_step=8,
        horizon=3,
        # hooks=dict(evaluate_pi=dict()),
        hooks=None,
        batch_size=256,
        buffer=dict(max_episode_num=200),
        _variants=dict(
        ),
        # cem = dict(seed_steps=1000),
    ) # do not know if we need max_grad_norm
    trainer.run_rpgm()
