import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
#from rpg.soft_rpg import Trainer
from rpg.skill import SkillLearning

env = GymVecEnv('Humanoid-v3', 1, ignore_truncated_done=True)
# env = GymVecEnv('Hopper-v3', 1, ignore_truncated_done=True)

trainer = SkillLearning.parse(
    env, buffer=dict(), head=dict(std_mode='statewise', std_scale=1., squash=True), optim=dict(max_grad_norm=1., lr=0.0001), update_train_step=1, have_done=True, hooks=dict(evaluate_pi=dict()), weights=dict(done=100., state=5000.), wandb=dict(name='humanoid-rpg'), qmode='Q', gamma=0.97,

    _variants = dict(
        gamma97_h3=dict(gamma=0.97, horizon=3),
        gamma98=dict(gamma=0.98, horizon=3),
        gamma98_h6=dict(gamma=0.98, horizon=6),
        rpg=dict(
            gamma=0.98, horizon=6,
            z_dim=0,
            z_cont_dim=10,
            info=dict(mutual_info_weight=0.001),
        ), 
    )
) # do not know if we need max_grad_norm
# gamma = 0.97
trainer.run_rpgm()
