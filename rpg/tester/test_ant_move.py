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
    z_cont_dim=5,
    steps_per_epoch=1000,
    buffer=dict(max_episode_num=20000),
    head=dict(
        std_mode='statewise',
        std_scale = 1.,
        squash=True
    ),
    enta=dict(coef=1., target_mode='auto', target=0.),
    entz=dict(coef=100000., target_mode='none'),
    optim=dict(max_grad_norm=1., lr=0.0003),
    horizon=3,
    actor_delay=2, #10,

    update_train_step=1,
    hooks=dict(save_traj=dict()),
    path='tmp/ant',
    weights=dict(reward=100., q_value=1.),

    info=dict(mutual_info_weight=3., action_weight=1., obs_weight=1., epsilon=0.01),

    # gamma=0.97,

    eval_episode=10,
    save_video=300, # save video ..
    batch_size=512,
    qmode='Q',
    _variants=dict(
        normal=dict(info=dict(mutual_info_weight=3.), enta=dict(target=-4.), wandb=dict(name='normal_ant')),
        smallent=dict(info=dict(mutual_info_weight=10.), enta=dict(target=-8.), wandb=dict(name='normal_ant')),
        smallent2=dict(info=dict(mutual_info_weight=5.), enta=dict(target=-8.), wandb=dict(name='normal_ant')),

        small_reward=dict(info=dict(mutual_info_weight=1.), ir=dict(reward_decay=dict(init_value=0.2)), wandb=dict(name='normal_ant', stop=True)),
        small_reward2=dict(info=dict(mutual_info_weight=1.), ir=dict(reward_decay=dict(init_value=0.4)), wandb=dict(name='normal_ant')),

        mbrl=dict(z_dim=1, z_cont_dim=0),

        maxent0=dict(z_dim=0, z_cont_dim=4, info=dict(mutual_info_weight=2.), ir=dict(reward_decay=dict(init_value=1.)), wandb=dict(name='antmove')),
        maxent1=dict(z_dim=0, z_cont_dim=4, info=dict(mutual_info_weight=2.), ir=dict(reward_decay=dict(init_value=0.25)), wandb=dict(name='antmove')),
        maxent2=dict(z_dim=0, z_cont_dim=4, info=dict(mutual_info_weight=2.), ir=dict(reward_decay=dict(init_value=0.5)), wandb=dict(name='antmove')),

        maxent3=dict(z_dim=0, z_cont_dim=4, info=dict(mutual_info_weight=2.), ir=dict(reward_decay=dict(init_value=4.)), wandb=dict(name='antmove')),


        norew=dict(z_dim=0, z_cont_dim=4, info=dict(mutual_info_weight=1., action_weight=0.), ir=dict(reward_decay=dict(init_value=0.02)), wandb=dict(name='antmove'), enta=dict(coef=1., target=2.)),

        norew2=dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.2, action_weight=0.), ir=dict(reward_decay=dict(init_value=0.01)), wandb=dict(name='antmove'), enta=dict(coef=1., target=2.)),

        rew = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.2, action_weight=0.), ir=dict(reward_decay=dict(init_value=0.1)), wandb=dict(name='antmove'), enta=dict(coef=1., target=2.)),

        rew_incr = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.2, action_weight=0.), ir=dict(reward_decay=dict(init_value=0.1, TYPE='exp', start=40000, end=200000, target_value=1.)), wandb=dict(name='antmove'), enta=dict(coef=1., target=1.)),

        rew2 = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.2, action_weight=0.), ir=dict(reward_decay=dict(init_value=0.1), info_decay=dict(init_value=1., TYPE='exp', start=50000, end=500000, target_value=0.1)), wandb=dict(name='antmove'), enta=dict(coef=1., target=2.)),

        infodecay = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.2, action_weight=0.), ir=dict(info_decay=dict(init_value=1., TYPE='exp', start=100000, end=500000, target_value=0.1)), wandb=dict(name='antmove'), enta=dict(coef=1., target=2.)),

        discrete=dict(z_dim=10, z_cont_dim=0, info=dict(mutual_info_weight=3.), entz=dict(coef=100.), ir=dict(entz_decay=dict(TYPE='linear', start=50000, end=100000, target_value=0.)), wandb=dict(name='antmove')),

        discrete2=dict(z_dim=10, z_cont_dim=0, info=dict(mutual_info_weight=30.), entz=dict(coef=100.), ir=dict(entz_decay=dict(TYPE='linear', start=50000, end=100000, target_value=0.)), wandb=dict(name='antmove')),

        discrete3=dict(z_dim=10, z_cont_dim=0, info=dict(mutual_info_weight=300.), entz=dict(coef=100.), ir=dict(entz_decay=dict(TYPE='linear', start=50000, end=100000, target_value=0.)), wandb=dict(name='antmove')),

        discrete4=dict(z_dim=10, z_cont_dim=0, info=dict(mutual_info_weight=30.), entz=dict(coef=100.), ir=dict(entz_decay=dict(TYPE='linear', start=50000, end=100000, target_value=0.), reward_decay=dict(init_value=0.3)), wandb=dict(name='antmove')),

        cont = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=1.), entz=dict(coef=20.), ir=dict(entz_decay=dict(init_value=1., TYPE='linear', start=100000, end=200000, target_value=0.)), wandb=dict(name='antmove'), enta=dict(coef=1., target=0.), 
                    pi_z=dict(head=dict(std_mode='fix_learnable', std_scale=1., nocenter=False, squash=True, linear=False))
                    ),
        cont2 = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=1.), entz=dict(coef=20.), ir=dict(entz_decay=dict(init_value=1., TYPE='linear', start=100000, end=200000, target_value=0.)), wandb=dict(name='antmove'), enta=dict(coef=1., target=1.), 
                    pi_z=dict(head=dict(std_mode='fix_learnable', std_scale=1., nocenter=False, squash=True, linear=False))),

        cont3 = dict(z_dim=0, z_cont_dim=12, info=dict(mutual_info_weight=0.1), entz=dict(coef=20.), ir=dict(entz_decay=dict(init_value=1., TYPE='linear', start=20000, end=500000, target_value=0.)), wandb=dict(name='antmove'), enta=dict(coef=1., target=0.), 
                    pi_z=dict(head=dict(std_mode='fix_learnable', std_scale=1., nocenter=False, squash=True, linear=False))),
    )
) # do not know if we need max_grad_norm
trainer.run_rpgm()
