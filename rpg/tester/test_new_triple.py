import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer


N = 100
env = TorchEnv('TripleMove', N, ignore_truncated_done=True , n_goals=3, goal_type=1)
# 150000

trainer = Trainer.parse(
    env, 
    steps_per_epoch=150,

    buffer=dict(max_episode_num=100000),
    #z_dim=6,
    #z_cont_dim=0,
    #hidden=dict(n=6),
    hidden=dict(),

    model=dict(qmode='value'),
    update_train_step=1,
    horizon=3,
    actor_delay=4, #10,
    z_delay=4,


    trainer=dict(weights=dict(reward=10000., q_value=100., state=1000.)),
    pi_a=dict(ent=dict(coef=0.005),),
    head=dict(
        linear=False,
        squash=True,
        std_mode='fix_no_grad',
        std_scale = 0.1
    ),
    pi_z=dict(ent=dict(coef=1000., target_mode='none')),

    path='tmp/new',
    hooks=dict(save_traj=dict(n_epoch=4, save_gif_epochs=10)),
    info=dict(coef=0.0),
    # info=dict(mutual_info_weight=0.03, action_weight=1., obs_weight=1., epsilon=0.01, std_mode='fix_no_grad'),

    _variants=dict(
        sac=dict(model=dict(qmode='Q'), horizon=1, trainer=dict(weights=dict(state=0.))),
        value=dict(model=dict(qmode='value'), horizon=1),
        sac3=dict(model=dict(qmode='Q'), horizon=3),
        value3=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.2)),
        value3_2=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.1, squash=False)),

        #z=dict(model=dict(qmode='value'), horizon=3, z_dim=6, info=dict(coef=1.), pi_a=dict(pi=dict(head=dict(std_scale=0.1)))),
        z = dict(_inherit='value3', hidden=dict(n=6), info=dict(coef=0.1)),
        z2=dict(_inherit='z', head=dict(std_scale=0.05)),
        relabelz=dict(_inherit='z2', relabel=0.8),

        
        normal=dict(
            model=dict(qmode='value'), horizon=3,
            hidden=dict(TYPE='Gaussian', dim=5), 
            info=dict(coef=0.01, weight=dict(TYPE='linear', min_value=0.001, end=4000)),
            head=dict(std_scale=0.1),
            path='tmp/normal'
        )
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
