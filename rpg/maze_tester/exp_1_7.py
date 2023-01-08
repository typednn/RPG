from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antfork2',
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
                n=[12, 1, 12, 12, 12]
            ),
            info=dict(coef=[0.03, 0.0, 0.03, 0.01, 0.005]),
            env_cfg=dict(n=[1, 1, 5, 1, 1]),
            #save_video=100,
        ),
        base='antcross', default_env='AntFork',
        names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005']
    )


    exp.main()