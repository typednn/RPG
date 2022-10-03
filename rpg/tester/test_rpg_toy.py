import tqdm
import numpy as np
from nn.space import Discrete
from rpg.env_base import TorchEnv
from rpg.rpg import train_rpg

N = 10
env = TorchEnv('TripleMove', N)
hidden_space  = Discrete(4)

train_rpg(env, hidden_space, steps=env.max_time_steps, hidden_head=dict(epsilon=0.))