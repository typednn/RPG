import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.skill import SkillLearning

# max_grad_norm=1.
env = GymVecEnv('HalfCheetah-v3', 1, ignore_truncated_done=True)
trainer = SkillLearning.parse(
    env, 
    head=dict(std_mode='statewise', std_scale=1., squash=True), 
    enta=dict(coef=1., target=-6), 
    optim=dict(max_grad_norm=1.),
    horizon=3,
    update_train_step=1,
    hooks=dict(evaluate_pi=dict()),
    z_dim=0,
    z_cont_dim=10,

    wandb=dict(name='cheetah-rpg'),
    info=dict(mutual_info_weight=0.01),
    _variants=dict(
        mbrl=dict(z_cont_dim=0, z_dim=1),
    ),
) # do not know if we need max_grad_norm
trainer.run_rpgm()
