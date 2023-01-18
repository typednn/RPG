from rpg.maze_tester.maze_exp import *

configs = dict(
    env_name = dict(
        cabinet='Cabinet',
        ant='AntPush'
    )
)

if __name__ == '__main__':
    exp = build_exp(base_config)

    for env_name  in ['cabinet', 'ant']: # ensure the experiments are finished ..
        exp.add_exps(
            f'dense{env_name}',
            dict(
                env_cfg=dict(n=5, reward_type='dense', obs_dim=0),
                _base=['mbsacv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgcv2', 'rewardrpg', 'rewardrpgc'],
                info=dict(coef=[0.0, 0.5, 1., 5., 10., 50., 1., 1., 1.]),
                rnd=dict(scale=0.),
                z_delay=4,
                pi_a=dict(ent=dict(coef=1.)),
                pi_z=dict(ent=dict(coef=100.)),
                reward_schedule="1000000" if env_name == 'ant' else "100000",
            ),
            names=['sac', 'discrete5', 'discrete10', 'discrete50', 'discrete100', 'discrete500', 'gaussian10', 'incrR', 'incrRc1'],
            base='mbsacv3', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()