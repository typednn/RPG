import matplotlib.pyplot as plt
from solver.draw_utils import draw_np
from solver.envs import TrajBandit, FourBandit

env = FourBandit()
env.reset()
img = env.render('rgb_array')
plt.imshow(img/255)
plt.savefig('y.png')


import numpy as np
images = []
for i in range(50):
    action = np.random.random(size=(1,2))*2-1
    #action[[0]j = 1.
    #action[[4]] = 1.
    #action[6] = -1.
    obs, r, _, info = env.step(action)
    images.append(env.render('rgb_array'))
from tools.utils import animate
animate(images)