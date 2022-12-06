import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer

# max_grad_norm=1.

N = 100
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=3, goal_type=1)

trainer = Trainer.parse(
    env, 
    steps_per_epoch=150,

    z_dim=6,
    z_cont_dim=0,

    model=dict(qmode='value'),
    update_train_step=1,
    horizon=3,
    actor_delay=4, #10,
    z_delay=4,

    hooks=dict(save_traj=dict(n_epoch=4, save_gif_epochs=10)),
    path='tmp/new',

    trainer=dict(
        weights=dict(reward=1000., q_value=100.)
    ),

    pi_a=dict(
        ent=dict(coef=0.02),
    ),
    pi_z=dict(
        ent=dict(coef=10., target_mode='none'),
    ),

    # info=dict(mutual_info_weight=0.03, action_weight=1., obs_weight=1., epsilon=0.01, std_mode='fix_no_grad'),

    _variants=dict(
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
