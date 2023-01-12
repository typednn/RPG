from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)


    exp.add_exps(
        'block',
        dict(
            _base=['rpgnormal', 'mbsacrnd5'],
            env_cfg=dict(reward_type='sparse', n_block=2),
        ),
        base=None, default_env='BlockPush',
    )

    exp.add_exps(
        'stickpull',
        dict(
            _base=['rpgnormal', 'mbsacrnd5'],
            env_cfg=dict(reward_type='sparse'),
        ),
        base=None, default_env='MWStickPull',
    )

    exp.main()