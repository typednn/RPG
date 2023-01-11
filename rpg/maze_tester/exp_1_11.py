from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'hammer',
        dict(
            _base=['mbsac', 'mbsacrnd', 'rpgnormal', 'rpgdiscrete'],
            env_cfg=dict(reward_type='sparse'),
        ),
        base=None, default_env='AdroitHammer',
        #names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )


    exp.main()