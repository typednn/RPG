import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer
#from rpg.oc import OptionCritic
from rpg.mutlimodal import MultiModal

# max_grad_norm=1.
N = 10
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=3, boundary=True)

trainer = MultiModal.parse(
    env, max_update_step=0,
    steps_per_epoch=200,
    buffer=dict(max_episode_num=20000),
    head=dict(
        # std_mode='statewise',
        #std_mode = 'statewise',
        std_mode='fix_no_grad',
        std_scale=0.08,
        #std_scale = 0.2,
        squash=False,
        # nocenter=True,
    ),
    z_head=dict(epsilon=0.0),
    z_cont_dim=0,
    z_dim=12,

    qnet=dict(gamma=0.,),

    entropy_coef=0.0,
    entropy_target=-2.,
    actor_optim=dict(max_grad_norm=1., lr=0.0003),
    #horizon=3,
    horizon=1,
    actor_delay=10, #10,
    # tau = 0.001,
    # tau=0.001,
    update_train_step=1,
    hooks=dict(save_traj=dict(n_epoch=1, save_gif_epochs=100)),
    #path='tmp/multimodal',
    # weights=dict(prefix=1.),
    pg=False,

    #entz_coef=0.05,
    #entz_coef=0.0,
    # entz_coef=0.01,
    entz_coef = 0.1,
    # entz_coef = 0.1,
    entz_target = 1.5,
    ir=dict(mutual_info_weight=1., action_weight=1., obs_weight=1.),
    weights=dict(prefix=1000.),

    eval_episode=1,
    batch_size=512,
    z_grad=False,


    ppo = 0,
    warmup_steps=2000,

    info_lr=0.003,
    actor_mode = 'soft',
    path='tmp/soft',
    # actor_mode = 'discrete',
    # path='tmp/discrete',
) # do not know if we need max_grad_norm
trainer.run_rpgm()
