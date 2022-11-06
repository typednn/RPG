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

import gym
from gym.spaces import Box, Discrete
from .utils import get_maze_env_obs, render_maze, render_background
from .maze import MazeGame

class PacManEnvSingleLayer(gym.Env):
    def __init__(self, height, width,
                 reset_maze = True,
                 reset_goal = True,
                 block_size=20, penalty=0):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.observation_space = Box(-np.inf, np.inf, (6, height, width))
        self.action_space = Discrete(5)
        self.penalty = penalty

        self.reset_maze = reset_maze
        self.reset_goal = reset_goal
        self.maze = None
        import random
        state = random.getstate()
        random.seed(0)
        self.reset()
        random.setstate(state)

    def get_obs(self):
        obs = self._high_background.copy()
        x, y = map(int, self.loc)
        obs[4, y, x] = 1

        x, y = map(int, self.goal)
        obs[5, y, x] = 1
        return obs

    def reset(self):
        if self.reset_maze or self.maze is None:
            self.maze = MazeGame(self.height, width=self.width)
        self.maze.reset(reset_target=self.reset_goal)

        self._background = None
        self._high_background = get_maze_env_obs(self.maze)
        self.loc = np.array(self.maze.player) + 0.5
        self.goal = np.array(self.maze.target) + 0.5
        return self.get_obs()

    def get_reward(self):
        loc = tuple(map(int, self.loc))
        goal = tuple(map(int, self.goal))
        reward = int(loc == goal)
        return reward


    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            action = action[0]
        if isinstance(action, np.ndarray) or isinstance(action, list):
            action = action[0]
        next = (x, y) = self.loc
        cell = self.maze.maze[int(x), int(y)]
        penalty = 0
        if action == 0:
            if 'n' not in cell:
                next = (x, y - 1)
            else:
                penalty = 1
        if action == 1:
            if  's' not in cell:
                next = (x, y + 1)
            else:
                penalty = 1
        if action == 2:
            if 'w' not in cell:
                next = (x - 1, y)
            else:
                penalty = 1
        if  action == 3:
            if 'e' not in cell:
                next = (x + 1, y)
            else:
                penalty = 1
        self.loc = np.array(next)
        rewards = self.get_reward()
        return self.get_obs(), rewards - penalty * self.penalty, False, {'success': rewards > 0.5, "done_bool": False}

    def render(self, mode):
        if self._background is None and mode == 'rgb_array':
            self._background = render_background(self.maze, self.block_size)
        return render_maze(self.maze, self.block_size, self._background, None, self.goal, self.loc, mode)

