import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer
#from rpg.oc import OptionCritic

# max_grad_norm=1.
N = 1
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=3)

trainer = Trainer.parse(
    env,
    pi_z=dict(K=1, epsilon=0.2),
    z_dim=10,
    steps_per_epoch=150,
    buffer=dict(max_episode_num=20000),
    head=dict(
        std_mode='fix_no_grad',
        std_scale=0.2,
        squash=False
        # std_mode='statewise',
        # std_scale = 0.1,
        # squash=True
    ),
    gamma=0.9,
    enta=dict(coef=0.0, target=-0.5),
    #enta = dict(coef=0.),
    entz=dict(coef=0.008, target=1.4),

    optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=3,
    z_delay=2,
    actor_delay=10, #10,

    update_train_step=1,
    hooks=dict(save_traj=dict(n_epoch=4, save_gif_epochs=10)),
    path='tmp/oc',
    weights=dict(reward=1000., q_value=100.),

    info=dict(mutual_info_weight=0.1, action_weight=1., obs_weight=1., epsilon=0.01),

    eval_episode=10,
    batch_size=512,
    qmode='value',
) # do not know if we need max_grad_norm
trainer.run_rpgm()
