from rpg.maze_tester.maze_exp import *

def add_var(k, d):
    assert k not in base_config['_variants']
    base_config['_variants'][k] = d

    
if __name__ == '__main__':
    exp = build_exp(base_config)

    # search for suitable RND value first
    exp.add_exps(
        'entrnd', dict(rnd=dict(scale=[1., 10., 100., 1000.],)), 
        base='ant_maxent', default_env='AntMaze2',
    )

    # TODO: test various representation:
    #  - normal RL - discrete - gaussian - uniform

    # TODO: consider harder env.

    # TODO: consider adding reward back
    exp.main()