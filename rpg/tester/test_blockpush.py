import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo
from rpg.soft_rpg import Trainer
#from rpg.oc import OptionCritic

# max_grad_norm=1.
N = 1
env = GymVecEnv('BlockPush', N, ignore_truncated_done=True)

trainer = Trainer.parse(
    env,
    pi_z=dict(K=100000),
    z_dim=1,
    steps_per_epoch=1000,
    buffer=dict(max_episode_num=20000),
    head=dict(
        std_mode='statewise',
        std_scale = 0.3,
        squash=True
    ),
    enta=dict(coef=1., target_mode='auto'),
    entz=dict(coef=4., target_mode='none'),
    optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=2,
    actor_delay=4, #10,

    update_train_step=1,
    hooks=dict(evaluate_pi=dict()),
    path='tmp/blockpush',
    weights=dict(reward=1000., q_value=100.),

    info=dict(mutual_info_weight=0.1, action_weight=1., obs_weight=1., epsilon=0.01),

    eval_episode=1,
    save_video=300, # save video ..
    batch_size=512,
    qmode='value',
) # do not know if we need max_grad_norm
trainer.run_rpgm()
