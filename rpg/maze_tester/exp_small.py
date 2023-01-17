
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    
    for name in ['reward', 'explore']:
        exp.add_exps(
            f'small{name}',
            dict(
                # TODO: action_scale=0.3
                reward_scale=float((name=='reward'))*5.,
                env_cfg=dict(n=5),
                max_epoch=1000,
                _base=['rpgdv3', 'mbsacv3', 'rewardrpg', 'rewardsac', 'mpc'],
                save_video=0,
                pi_z=dict(ent=dict(coef=10., target_mode='none')),
                info=dict(coef=0.1),
                reward_schedule='100000',
                z_delay=10,
            ),
            base=None, default_env='SmallMaze2',
        )


    exp.main()