from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'small',
        dict(
            reward_scale=0.,
            env_cfg=dict(n=1),
            max_epoch=40,
            rnd=dict(
                density=dict(
                    TYPE='RND',
                    normalizer='ema',
                ),
            ),
            hidden=dict(n=[1, 6, 1, 6, 1, 6, 1, 6]),
            pi_a=dict(ent=dict(coef=[0.01, 0.01, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01])),
            head=dict(std_scale=[0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 1., 1.]),
        ),
        base='small', default_env='SmallMaze',
        names=['rl', 'rpg', 'rlnoenta', 'rpgnoenta', 'rl04', 'rpg04', 'rl10', 'rpg10']
    )

    exp.add_exps(
        'small2',
        dict(
            reward_scale=0.,
            env_cfg=dict(n=1, action_scale=0.3),
            max_epoch=100,
            rnd=dict(
                density=dict(
                    TYPE='RND',
                    normalizer='ema',
                ),
            ),
            hidden=dict(n=[1, 6, 1, 6, 1, 6, 1, 6]),
            pi_a=dict(ent=dict(coef=[0.01, 0.01, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01])),
            head=dict(std_scale=[0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 1., 1.]),
        ),
        base='small', default_env='SmallMaze',
        names=['rl', 'rpg', 'rlnoenta', 'rpgnoenta', 'rl04', 'rpg04', 'rl10', 'rpg10']
    )

    exp.add_exps(
        'antcross4',
        dict(
            reward_scale=0.,
            rnd=dict(
                density=dict(
                    TYPE='RND',
                    normalizer='ema',
                ),
                scale = 0.1,
            ),
            hidden=dict(
                TYPE='Categorical',
                n=[12, 12]
            ),
            info=dict(coef=[0.05, 0.1]),
        ),
        base='antcross', default_env='AntMaze3',
        names=['rnd12_5', 'rnd12_10']
    )

    exp.main()