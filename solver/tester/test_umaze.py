from solver.draw_utils import draw_np
from solver.envs import SparseGripper, SimpleStartGripper, FasterGripper, FixMove, HardPush

env = HardPush()
env.reset_state_goal(*env.reset())

import numpy as np
images = []
for i in range(50):
    action = np.random.random(size=6)*2-1
    #action[[0]] = 1.
    #action[[4]] = 1.
    #action[6] = -1.
    obs, r, _, info = env.step(action)
    images.append(env.render('rgb_array'))
from tools.utils import animate


print(obs['pointcloud'].shape)
animate(images)