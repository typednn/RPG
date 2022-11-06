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

from hrl.envs.hiro_robot_envs.maze_env_v2 import MazeEnvV2
from hrl.envs.hiro_robot_envs.ant import AntEnv
from hrl.envs import make

class NewAntEnv(AntEnv):
    def viewer_setup(self):
        self.viewer.cam.lookat[0] += 3.5
        self.viewer.cam.lookat[1] += 3.5
        self.viewer.cam.lookat[2] = 5

        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)

env = make('PacMan-v1', None)
message = env.reset()
print(message.state)

class AntMazeEnv(MazeEnvV2):
    MODEL_CLASS = NewAntEnv

#env = create_maze_env(env_name='AntMaze')
env = AntMazeEnv(maze_size_scaling=4, wall_size=0.1)

env.reset()
imgs = []
for i in range(4):
    imgs.append(env.render(mode='rgb_array'))
    #cv2.imshow('x', imgs[-1][...,::-1])
    #cv2.waitKey(1)
    env.step(env.action_space.sample())

#plt.imshow(np.concatenate(imgs, 1))
#plt.show()

#%%
name = "wall_0_0_h"
id = env.wrapped_env.model.geom_name2id(name)
print(env.wrapped_env.model.geom_pos[id], )

#%%
# env.wrapped_env.sim.data.geom_xpos + 100
print(env.wrapped_env.model.geom_pos)
env.wrapped_env.model.geom_pos[:10] += 10
env.wrapped_env.sim.forward()
#print(env.wrapped_env.sim.data.geom_xpos[0])

plt.imshow(env.render(mode='rgb_array'))
plt.show()
env.wrapped_env.model.geom_pos[:10] -= 10
env.wrapped_env.sim.forward()
