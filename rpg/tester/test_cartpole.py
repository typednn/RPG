import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.ppo import train_ppo
from rpg.qlearning import QTrainer

# max_grad_norm=1.
env = GymVecEnv('CartPole-v1', 1, ignore_truncated_done=True)
trainer = QTrainer.parse(
    env, 
    head=dict(std_mode='statewise', std_scale=1., squash=True), 
    enta=dict(coef=1., target=-6), 
    optim=dict(max_grad_norm=1.),
    horizon=3,
    update_train_step=1,
    hooks=dict(evaluate_pi=dict()),
    have_done=True,
    # wandb=dict(name='cheetah'),
    qmode='value',
) # do not know if we need max_grad_norm
trainer.run_rpgm()
