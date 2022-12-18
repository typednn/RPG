"""
experiment template; needs to specify
  naming: 
    env_name + variant module (base, can be omited) + ablation factor
        - we can register base variant names
        - ablation factors, we can register its types and so on
  algorithm variants: basis
  factors:
        - 
  envs 
  metric 
  ways of saving and visualization
"""
import copy
from tools.config import Configurable, CN, merge_a_into_b, extract_variant


base_config = CN(
    max_epochs=1000, # 200 * 5 * 1000
    steps_per_epoch=200,
    env_name='SmallMaze',
    env_cfg=dict(n=5, ignore_truncated_done=True, reward=False),
    buffer=dict(max_episode_num=100000),
    #z_dim=6,
    #z_cont_dim=0,
    hidden=dict(n=6),

    model=dict(qmode='value'),
    update_train_step=1,
    horizon=3,
    actor_delay=4, #10,
    z_delay=4,


    trainer=dict(weights=dict(reward=10000., q_value=100., state=1000.)),
    pi_a=dict(ent=dict(coef=0.0, target_mode='fixed'),),
    head=dict(
            linear=False,
            squash=True,
            std_mode='fix_no_grad',
            std_scale = 0.2
    ),
    pi_z=dict(ent=dict(coef=1000., target_mode='none'),),

    path='tmp/maze',
    hooks=dict(save_traj=dict(n_epoch=4, save_gif_epochs=10, occupancy=1)),
    info=dict(coef=0.0),
    # info=dict(mutual_info_weight=0.03, action_weight=1., obs_weight=1., epsilon=0.01, std_mode='fix_no_grad'),

    _variants=dict(
        sac=dict(model=dict(qmode='Q'), horizon=1, trainer=dict(weights=dict(state=0.))),
        value=dict(model=dict(qmode='value'), horizon=1),
        sac3=dict(model=dict(qmode='Q'), horizon=3),
        value3=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.2)),
        value3_2=dict(model=dict(qmode='value'), horizon=3, hidden=dict(n=1), head=dict(std_scale=0.1, squash=False)),
        #z=dict(model=dict(qmode='value'), horizon=3, z_dim=6, info=dict(coef=1.), pi_a=dict(pi=dict(head=dict(std_scale=0.1)))),
        z = dict(_inherit='value3', hidden=dict(n=6), info=dict(coef=0.1)),
        z2=dict(_inherit='z', head=dict(std_scale=0.05)),
        relabelz=dict(_inherit='z2', relabel=0.8),

        rnd = dict(
            _inherit='z2',
            rnd=dict(scale=1.),
        ),
        rnd2=dict(_inherit='rnd', head=dict(std_scale=0.3, std_mode='statewise')),
        # medium=dict(_inherit='rnd2', env_name='MediumMaze'),
        medium2=dict(_inherit='rnd2', env_name='MediumMaze', head=dict(std_scale=0.2, std_mode='fix_no_grad', linear=False, squash=False), pi_a=dict(ent=dict(coef=0.01)), rnd=dict(scale=1.), info=dict(coef=0.1), path='tmp/medium'), # seems that we can continue to decrease the info coef
        medium0=dict(_inherit='medium2', z_dim=1, path='tmp/medium0'),
        lessinfo=dict(_inherit='medium2', info=dict(coef=0.02), path='tmp/lessinfo'),

        rndreward=dict(_inherit='medium2', rnd=dict(
            as_reward=True,
            training_on_rollout=False,
            obs_mode='obs',
            scale=0.1,
        ),
        env_cfg=dict(obs_dim=5),
        path='tmp/rndreward'),

        small=dict(_inherit='rndreward', env_cfg=dict(n=5), path='tmp/small2'),
    ),
)



class Experiment(Configurable):
    def __init__(self, cfg=None, env=None, variant=None, path='tmp', wandb=False) -> None:
        self.base_config = base_config
        self.env_configs = self.get_env_configs()
        self.wandb = wandb

    def get_variants(self):
        base = self.base_config._variants
        return base

    def get_env_configs(self):
        # name_name, env_config
        return {
            'SmallMaze': dict(n=5),
        }


    def setup_logger(self, env_name, base, factor, variants):
        # naming
        # path
        assert len(variants) == 1

    def build_configs(self, env_name, expname, cfgs, base=None):
        """
        base
        To specify the variants, also use dictionary, but for certain keys it can be a list. We only support list of the same size.
            - {
                **config
                _default: None
                _names: [] a list of names
              }
              then zip them together
        """
        names = cfgs.pop('_names', [])
        rename = len(names) > 0
        factor_name = []
        default = {}

        variants = []
        def set_keyval(d, keys, val):
            for k in keys[:-1]:
                if 'k' not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = val

        def process_variants(configs, keys):
            for k, v in configs:
                new_keys = keys + k.split('.')
                if isinstance(v, list):
                    factor_name.append(k)
                    if len(variants) == 0:
                        for i in range(len(v)):
                            variants.append(copy.deepcopy(default))
                    if not rename:
                        assert len(variants) == len(names), "The length of the list should be the same."
                    else:
                        if len(names) == 0:
                            names = [[] for i in range(len(v))]
                        for i in range(len(v)):
                            names[i].append(str(v[i]))
                    assert len(variants) == len(v), "The length of the list should be the same."
                    for i in range(len(v)):
                        set_keyval(variants[i], new_keys, v[i])
                elif isinstance(v, dict):
                    process_variants(v)
                else:
                    if len(variants) == 0:
                        set_keyval(default, new_keys, v)
                    else:
                        for i in range(len(variants)):
                            set_keyval(variants[i], new_keys, v)
                            
        process_variants(cfgs, [])
        factor_name = '_'.join(factor_name)
        names = [factor_name + '_'.join(n) for n in names]


        cfg = self.base_config.clone()
        cfg.defrost()

        if base is not None:
            var = extract_variant(cfg, self.get_variants())
            merge_a_into_b(var, cfg)
            

        outputs = []
        env_cfg = self.env_configs[env_name]

        for name, k in zip(names, variants):
            k =CN(k)
            var_cfg = cfg.clone()
            merge_a_into_b(k, var_cfg)
            merge_a_into_b(
                CN(env_name=env_name, env_cfg=env_cfg), var_cfg
            )
            cfg_name = f'{env_name}_{expname}_{name}'

            if self.wandb:
                var_cfg.wandb = {'project': self.path, 'name': cfg_name, 'group': expname}
            else:
                var_cfg.path = self.path 

            outputs.append(var_cfg)

        return outputs
        


    def run_config(self, cfg):
        from rpg.soft_rpg import Trainer
        trainer = Trainer(cfg=cfg)
        trainer.run_rpgm()