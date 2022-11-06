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


from gym.spaces import Discrete, Box
import numpy as np

class MixtureSpace(Box):
    def __init__(self, n, low, high, shape, nums=None):
        self.n = n
        self.category = Discrete(n)
        self.nums = nums
        super(MixtureSpace, self).__init__(low, high, shape)

    def seed(self, seed=None):
        out = super(MixtureSpace, self).seed(seed)
        self.category.seed(seed)
        return out

    def sample(self):
        return np.concatenate(([self.category.sample()], super(MixtureSpace, self).sample()))

    def contains(self, x):
        type = x[0]
        ac = x[1:]
        return self.category.contains(int(type)) and super(MixtureSpace, self).contains(ac)