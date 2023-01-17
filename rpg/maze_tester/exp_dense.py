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
                env_cfg=dict(n=5, reward_type='dense'),
                _base=['mbsacv3', 'rpgcv2', 'rpgcv2', 'rpgcv2', 'rpgcv2', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3'],
                info = dict(coef=[0.0, 0.002, 0.005, 0.001, 0.0005, 0.001, 0.005, 0.0005, 0.002])
            ),
            names=['sac', 'gaussian002', 'gaussian005', 'gaussian001', 'gaussian0005', 'discrete001', 'discrete005', 'discrete0005', 'discrete002'],
            base='mbsacv3', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()