import tqdm
import numpy as np
from rpg.env_base import TorchEnv
from rpg.ppo import train_ppo

N = 100
env = TorchEnv('TripleMove', N)

train_ppo.parse(
    env, steps=env.max_time_steps,
    ppo=dict(entropy=dict(coef=0.1, target=-1.), learning_epoch=2),

    actor=dict(head=dict(linear=True, std_scale=0.5, std_mode='statewise')),

    hooks=dict(save_traj=dict(n_epoch=10, save_gif_epochs=10),
                            # monitor_action_std=dict(n_epoch=10,),
                            log_info=dict(n_epoch=10))
)