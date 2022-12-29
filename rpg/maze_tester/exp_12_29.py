from rpg.maze_tester.maze_exp import *

def add_var(k, d):
    assert k not in base_config['_variants']
    base_config['_variants'][k] = d

    
if __name__ == '__main__':
    add_var(
        'ant_maxentrl', dict(
            _inherit='ant_squash', 
            hidden=dict(n=1), info=dict(coef=0.0),
            head=dict(std_scale=1.0, std_mode='statewise'),
            pi_a=dict(ent=dict(coef=1., target_mode='none')),
            rnd=dict(scale=1.),
        )
    )

    exp = build_exp(base_config)

    # search for pi_a coef
    exp.add_exps(
        'entcoef', dict(pi_a=dict(ent=dict(coef=[1., 0.1, 0.01, 0.001]),)), 
        base='ant_maxentrl', default_env='AntMaze2',
    )

    # search for suitable RND value first: reward scale, action scale + [info scale in the end]
    exp.add_exps(
        'entrnd', dict(rnd=dict(scale=[1., 10., 100., 1000.],)), 
        base='ant_maxent', default_env='AntMaze2',
    )

    # TODO: reduce the mutual info weight?
    # exp.add_exps(
    #     'entrnd', dict(rnd=dict(scale=[1., 10., 100., 1000.],)), 
    #     base='ant_maxent', default_env='AntMaze2',
    # )

    # TODO: test various representation:
    #  - normal RL - discrete - gaussian - uniform

    # TODO: consider harder env.

    # TODO: consider adding reward back
    exp.main()