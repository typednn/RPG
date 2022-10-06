import tqdm
import numpy as np
from nn.space import Discrete
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg

N = 100
env = TorchEnv('TripleMove', N)
hidden_space  = Discrete(5)

train_rpg.parse(env, hidden_space, steps=env.max_time_steps, hidden_head=dict(epsilon=0.), reward_norm=True, relbo=dict(prior=0.01, ent_a=0.02, ent_z=0.01), hooks=dict(save_traj=dict(n_epoch=1)))