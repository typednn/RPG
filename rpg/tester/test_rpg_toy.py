import tqdm
import numpy as np
from nn.space import Discrete
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg

N = 100
env = TorchEnv('TripleMove', N)
hidden_space  = Discrete(1)

train_rpg.parse(env, hidden_space, steps=env.max_time_steps, hidden_head=dict(epsilon=0.), reward_norm=True, relbo=dict(ent_z=0., ent_a=0., mutual_info=0., prior=0.))