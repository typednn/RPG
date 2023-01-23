from tools.config import CN, merge_a_into_b, extract_variant
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        f'gapexplore',
        dict(
            _variants=dict(
                _base = dict(
                    reward_scale=0., #float((name=='reward'))*5.,
                    env_cfg=dict(n=5),
                    max_epoch=1000,
                    save_video=0,
                    pi_z=dict(ent=dict(coef=10., target_mode='none')),
                    reward_schedule='40000',
                    z_delay=5,
                ),
                discirete=dict(
                    _base='rpgdv3',
                    info=dict(coef=0.1),
                ),
                gaussian=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                ),
                mbsac=dict(
                    _base='mbsacv3',
                ),
            )
        ), 
        base=None, default_env='GapMaze',
    )

    exp.main()