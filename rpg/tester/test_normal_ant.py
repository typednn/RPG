import tqdm
import numpy as np
from rpg.env_base import GymVecEnv
from rpg.ppo import train_ppo
#from rpg.soft_rpg import Trainer
#from rpg.oc import OptionCritic
from rpg.skill import SkillLearning 


# max_grad_norm=1.
N = 1
env = GymVecEnv('TripleAnt', N, ignore_truncated_done=True, n_goals=4)

trainer = SkillLearning.parse(
    env,
    pi_z=dict(K=100000),
    # z_dim=10,
    z_dim=0,
    z_cont_dim=4,
    steps_per_epoch=1000,
    buffer=dict(max_episode_num=20000),
    head=dict(
        std_mode='statewise',
        std_scale = 1.,
        squash=True
    ),
    enta=dict(coef=1., target_mode='auto', target=1.),
    entz=dict(coef=100000., target_mode='none'),
    optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=2,
    actor_delay=4, #10,

    update_train_step=1,
    hooks=dict(save_traj=dict()),
    path='tmp/tripleant_n',
    weights=dict(reward=100., q_value=1.),

    info=dict(mutual_info_weight=2., action_weight=1., obs_weight=1., epsilon=0.01),

    eval_episode=10,
    save_video=300, # save video ..
    batch_size=512,
    qmode='value',
) # do not know if we need max_grad_norm
trainer.run_rpgm()
