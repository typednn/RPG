from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'kitcheninfo',
        dict(
            #_base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='bonus', n=5),
            buffer=dict(max_episode_num=5000),
            info=dict(coef=[0.002, 0.005, 0.01, 0.0005, 0.0001, 0.0008])
        ),
        base='rpgcv2', default_env='Kitchen',
    )

    exp.add_exps(
        'kitchenreward',
        dict(
            #_base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='bonus', n=5),
            buffer=dict(max_episode_num=5000),
            #info=dict(coef=[0.002, 0.005, 0.01, 0.0005, 0.0001, 0.0008])
            reward_scale=[2.5, 1.],
        ),
        base='rpgcv2', default_env='Kitchen',
    )



    exp.add_exps(
        'antpush500',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
            buffer=dict(max_episode_num=4000),
        ),
        base=None, default_env='AntPush',
    )

    exp.main()