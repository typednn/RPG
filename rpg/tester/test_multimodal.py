import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.rpgm import Trainer
#from rpg.oc import OptionCritic
from rpg.mutlimodal import MultiModal

# max_grad_norm=1.
N = 10
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=1, boundary=False)

trainer = MultiModal.parse(
    env, max_update_step=0,
    steps_per_epoch=200,
    buffer=dict(max_episode_num=20000),
    head=dict(
        # std_mode='statewise',
        #std_mode = 'statewise',
        std_mode='fix_no_grad',
        std_scale=0.3,
        squash=False,
    ),
    entropy_coef=0.0,
    entropy_target=-2.,
    actor_optim=dict(max_grad_norm=None, lr=0.0003),
    horizon=3,
    actor_delay=2, #10,
    # tau = 0.001,
    # tau=0.001,
    update_train_step=1,
    hooks=dict(save_traj=dict(n_epoch=1, save_gif_epochs=10)),
    path='tmp/multimodal',
    # weights=dict(prefix=1.),
    pg=False,

    #entz_coef=0.05,
    #entz_coef=0.0,
    # entz_coef=0.01,
    entz_coef = 0.,
    entz_target = -2.,
    ir=dict(mutual_info_weight=0., action_weight=1., obs_weight=1.),
    weights=dict(prefix=1000.),

    eval_episode=1,
    batch_size=512,
    z_grad=False,

    ppo = 100,
) # do not know if we need max_grad_norm
trainer.run_rpgm()
