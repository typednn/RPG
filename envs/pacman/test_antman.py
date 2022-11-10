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

from hrl.envs.pacman.antman import AntManEnv

env = AntManEnv()

#%%
imgs = []
for random in [0, 1]:
    tmps = []
    for j in range(3):
        env.reset()
        tmps.append(env.render(mode='rgb_array'))
    imgs.append(np.concatenate(tmps, 1))
plt.imshow(np.concatenate(imgs, 0))
plt.show()