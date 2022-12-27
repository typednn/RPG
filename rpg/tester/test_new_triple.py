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
    z_delay=0,


    trainer=dict(weights=dict(reward=10000., q_value=100., state=1000.)),
    pi_a=dict(ent=dict(coef=0.0, target_mode='fixed'),),
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
        #relabelz=dict(_inherit='z2', relabel=0.8),

        
        normal=dict(
            model=dict(qmode='value'), horizon=3,
            hidden=dict(TYPE='Gaussian', n=5), 
            info=dict(coef=0.01, weight=dict(TYPE='linear', min_value=0.2, end=8000)),
            head=dict(std_scale=0.01),
            info_delay=2,
            path='tmp/normal'
        ),

        normal_relabel = dict(
            _inherit='normal',
            relabel=0.8,
            info=dict(coef=0.03, weight=dict(TYPE='linear', min_value=0.1, end=8000)),
            path='tmp/normal_relabel'
        ),

        goal=dict(
            time_embedding=10,
            _inherit='normal',
            hidden=dict(TYPE='Goal'),
            info=dict(coef=0.1, weight=dict(TYPE='linear', min_value=0.01, end=10000)),
            path='tmp/goal',
        ),

        discrete_goal=dict(
            time_embedding=10,
            _inherit='z2',
            hidden=dict(TYPE='DiscreteGoal', n=6),
            info=dict(coef=5.),
            path='tmp/goal_discrete',
        ),

        relabel=dict(
            _inherit='z2',
            relabel=0.8,
            path='tmp/relabel',
            info=dict(coef=0.1),
        ),

        optimz=dict(
            _inherit='z2',
            path='tmp/optimz',

            z_delay=8,
            pi_z=dict(ent=dict(coef=1000., target_mode='none', schedule=dict(TYPE='linear', min_value=0.00001, end=1000))),
        ),
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()