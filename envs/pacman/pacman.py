# a hierarchical pacman environment
import cv2
import gym
from gym.spaces import Box, Discrete
from .maze import MazeGame
import numpy as np
from .utils import get_maze_env_obs, render_maze, render_background
from tools.config import Configurable

class PacManEnv(Configurable, gym.Env):
    def __init__(self,
                 cfg=None,
                 height=4, width=4,
                 reset_maze=True,
                 reset_goal=True,
                 block_size=20,
                 reward_type='sparse',
                 action_scale=0.25,
                 include_low_obs=1., penalty=0.):

        super(PacManEnv, self).__init__(cfg)

        self.height = height
        self.width = width
        self.block_size = block_size
        self.action_scale = action_scale
        self.reward_type = reward_type
        self.include_low_obs = include_low_obs
        self.penalty = penalty
        self.wall_eps = 0.05

        self.observation_space = [Box(-np.inf, np.inf, (2+4+2,)),
                                  Box(-np.inf, np.inf, (6 + (2 if include_low_obs > 0 else 0), height, width)),]

        self.action_space = [Box(-1, 1, (2,)),Discrete(5)]

        self.reset_maze = reset_maze
        self.reset_goal = reset_goal
        self.maze = None
        import random
        state = random.getstate()
        random.seed(0)
        self.reset()
        random.setstate(state)

        self._max_episode_steps = max(height, width) * 20
        self._substeps = 5


    def get_obs(self):
        high = self._high_background.copy()

        x, y = map(int, self.loc)
        high[4, y, x] = 1

        _x, _y = map(int, self.goal)
        high[5, _y, _x] = 1


        subgoal = tuple(map(int, self.subgoal))
        diff = np.array(subgoal) + 0.5 - self.loc

        dx = self.loc - np.floor(self.loc)
        if self.include_low_obs != 0:
            high[6:8] = dx[:, None, None] * self.include_low_obs

        #return {
        #    'meta': high,
        #    'low': np.concatenate((dx * 0.01, high[y, x][:4] * 0, diff))
        #}
        return (np.concatenate((dx * 0.01, high[y, x][:4] * 0, diff)), high)


    def get_reward(self):
        loc = tuple(map(int, self.loc))
        goal = tuple(map(int, self.goal))
        subgoal = tuple(map(int, self.subgoal))
        high_reward = int(loc == goal)  - self._penalty * self.penalty
        low_success = int(loc == subgoal)
        if self.reward_type == 'sparse':
            low_reward = int(loc == subgoal)
        else:
            low_reward = -np.linalg.norm(self.loc - (np.float32(subgoal) + 0.5))

        return {
            'meta': high_reward,
            'low': low_reward,
            'low_success': low_success,
            'success': int(loc==goal)
        }

    def reset(self):
        if self.reset_maze or self.maze is None:
            self.maze = MazeGame(self.height, width=self.width)
        self.maze.reset(reset_target=self.reset_goal)

        self._background = None
        self._high_background = get_maze_env_obs(self.maze, self.observation_space[1].shape[0])


        self.rects = []
        for i in range(self.width):
            for j in range(self.height):
                cell = self.maze.maze[i, j]
                if 'n' in cell and j > 0:
                    self.rects.append([[i, j-self.wall_eps], [i+1, j+self.wall_eps]])
                if 'w' in cell and i > 0:
                    self.rects.append([[i-self.wall_eps, j], [i+self.wall_eps, j+1]])
        self.rects.append([[0, -self.wall_eps], [self.width, self.wall_eps]])
        self.rects.append([[0, self.height-self.wall_eps], [self.width, self.height + self.wall_eps]])
        self.rects.append([[-self.wall_eps, 0], [self.wall_eps, self.height+1]])
        self.rects.append([[self.width-self.wall_eps, 0], [self.width + self.wall_eps, self.height]])
        self.rects = np.array(self.rects)

        # double check by draw rects...
        self.loc = np.array(self.maze.player) + 0.5
        self.goal = np.array(self.maze.target) + 0.5
        self.subgoal = tuple(map(int, self.loc))
        return self.get_obs()

    def draw_rects(self):
        imgs = np.zeros((512, 512, 3))
        f = lambda x: (int(x[0]/self.width * 512), int(x[1]/self.height * 512))
        for i in self.rects:
            cv2.rectangle(imgs, f(i[0]), f(i[1]), (255, 0, 255), -1)
        return imgs

    def step_meta(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            action = np.array(action)
            assert action.size == 1
            action = action[0]
        assert action in self.action_space[1]
        x, y = self.loc.copy()
        x = int(x)
        y = int(y)
        # action, nswe
        cell = self.maze.maze[x, y]
        if action == 0:
            self._penalty = 'n' in cell
            self.subgoal = (x, y-1)
        elif action == 1:
            self._penalty = 's' in cell
            self.subgoal = (x, y+1)
        elif action == 2:
            self._penalty = 'w' in cell
            self.subgoal = (x - 1, y)
        elif action == 3:
            self._penalty = 'e' in cell
            self.subgoal = (x + 1, y)
        else:
            self._penalty = 0
            self.subgoal = (x, y)
        return self.get_obs()

    def step_clip(self, loc, dir):
        # eight directions, from north, clock wise..
        from .utils import step
        return step(self.rects, loc[0], loc[1], dir[0], dir[1])

    def step(self, action):
        #if action.dot(self.get_obs()['low'][-2:]) < -0.1:
        #    print(self.get_obs()['low'][-2:], action, np.array(self.subgoal)+0.5, self.loc)
        #    exit(0)
        action = np.array(action).clip(-1, 1) * self.action_scale
        self.loc = np.array(self.step_clip(self.loc, action))
        rewards = self.get_reward()
        low_success = rewards['low_success']
        del rewards['low_success']

        return self.get_obs(), (rewards['low'], rewards['meta']), False, {
            'done_bool': False,
            'low_done_bool': False,
            'low_success': low_success,
            "success": rewards['success']
        }

    def render(self, mode):
        if self._background is None and mode == 'rgb_array':
            self._background = render_background(self.maze, self.block_size)
        return render_maze(self.maze, self.block_size, self._background, self.loc, self.goal, self.subgoal, mode)
