from tools.config import CN, merge_a_into_b, extract_variant
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        f'denseantabl',
        dict(
            _variants=dict(
                _base = dict(
                    env_cfg=dict(n=5, reward_type='dense', obs_dim=0),
                    z_delay=4,
                    rnd=dict(scale=0.),

                    pi_a=dict(ent=dict(coef=1.)),
                    pi_z=dict(ent=dict(coef=100.)),
                ),
                seg3=dict(
                    _base='rewardrpg',
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
                largestd=dict(
                    _base='rewardrpg',
                    pi_a=dict(ent=dict(coef=1., target=-4.)),
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
                noepsilon=dict(
                    _base='rewardrpg',
                    hidden=dict(head=dict(epsilon=0.02)),
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
                seg3n1=dict(
                    _base='rewardrpg',
                    env_cfg=dict(n=1),
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
            )
        ),
        base=None, 
        default_env ='AntPush',
    )

    exp.main()