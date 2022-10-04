import tqdm
import numpy as np
from nn.space import Discrete
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg

N = 10
env = TorchEnv('TripleMove', N)
hidden_space  = Discrete(4)

train_rpg.parse(env, hidden_space, steps=env.max_time_steps * 4, hidden_head=dict(epsilon=0.), reward_norm=False, relbo=dict(ent_z=0., ent_a=0., mutual_info=0., prior=0.))