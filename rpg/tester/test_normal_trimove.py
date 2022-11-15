import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
# from rpg.soft_rpg import Trainer
from rpg.skill import SkillLearning
#from rpg.oc import OptionCritic

# max_grad_norm=1.
N = 1
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=3)

trainer = SkillLearning.parse(
    env,
    z_dim=0,
    z_cont_dim=4,
    steps_per_epoch=150,
    buffer=dict(max_episode_num=20000),
    head=dict(
        #std_mode='fix_no_grad',
        #std_scale=0.2,
        # squash=False
        std_mode='statewise',
        std_scale = 0.3,
        squash=True
    ),
    enta=dict(coef=0.02, target=-2.),
    entz=dict(coef=4., target_mode='none'),
    optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=3,
    actor_delay=4, #10,

    update_train_step=1,
    hooks=dict(save_traj=dict(n_epoch=4, save_gif_epochs=10)),
    path='tmp/normal',
    weights=dict(reward=1000., q_value=100.),

    info=dict(mutual_info_weight=0.003, action_weight=1., obs_weight=1., epsilon=0.01),

    eval_episode=10,
    batch_size=512,
    qmode='value',

     ir=dict(entz_decay=dict(TYPE='exp', start=10, end=20000, min_value=0.1)),
    pi_z=dict(head=dict(std_mode='fix_learnable', std_scale=1., nocenter=False, squash=True, linear=False)),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
