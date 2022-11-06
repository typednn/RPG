import os
import cv2
import numpy as np
import torch
import copy
import pickle
from typing import Optional, TypeVar, Type, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from hrl.envs import make
import random
random.seed(1)

envs = ['PacMan-v1']#, 'PacManDense-v1', 'PacManFixed-v1', 'PacManFixedDense-v1']
imgs = []
imgs2 = []
for i in envs:
    env = make(i, env_cfg=None)
    tmp = []
    for j in range(3):
        meta = env.reset()
        env.env._penalty = 0
        img = env.env.draw_rects()
        imgs2.append(img)
        old_r1 = env.env.get_reward()['low']
        message = env.step(0)
        tmp.append(env.render(mode='rgb_array'))
        old_r = env.env.get_reward()['low']
        for i in range(5):
            o = env.step([-1, 1])
        print(env.step(None).reward)
        tmp.append(env.render(mode='rgb_array'))
        print(meta.reward, old_r1, old_r, o.reward, o.state, o.info['success'])
    imgs.append(np.concatenate(tmp, 1))
img = np.concatenate(imgs)
print(img.shape)

#cv2.imwrite('x.png', np.concatenate(imgs2, 1))
plt.imshow(np.concatenate(imgs2, 1))
plt.show()
plt.imshow(img)
plt.show()

