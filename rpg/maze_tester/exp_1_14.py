from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'kitchenbonus',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='bonus', n=5),
            buffer=dict(max_episode_num=5000),
        ),
        base=None, default_env='Kitchen',
    )



    exp.add_exps(
        'antpush2',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
        ),
        base=None, default_env='AntPush',
    )

    exp.main()