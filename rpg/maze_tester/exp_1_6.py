from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antfork',
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
                n=[12, 1]
            ),
            info=dict(coef=[0.03, 0.0]),
        ),
        base='antcross', default_env='AntFork',
        names=['rnd', 'rl']
    )

    # exp.add_exps(
    #     'blocktry',
    #     dict(
    #         save_video=300,
    #         steps_per_epoch=120, 
    #         hooks=dict(save_traj=dict(n_epoch=20)),
    #         env_cfg=dict(
    #             n_block=[1, 1, 2, 2, 3, 3, 1, 2, 3]
    #         ),
    #         hidden=dict(
    #             TYPE='Categorical',
    #             n=[12, 1, 12, 1, 12, 1, 6, 6, 6]
    #         ),
    #         info=dict(coef=0.01),
    #     ),
    #     base='block', default_env='BlockPush',
    # )

    exp.main()