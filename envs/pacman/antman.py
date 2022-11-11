import numpy as np
from tools.config import Configurable
from .pacman import PacManEnv, Box, Discrete, MazeGame, get_maze_env_obs, render_maze, render_background

from ..hiro_robot_envs.maze_env_v2 import MazeEnvV2
from ..hiro_robot_envs.ant import AntEnv

class NewAntEnv(AntEnv):
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        #from mujoco_py.generated import const
        #self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.lookat[0] = 9
        self.viewer.cam.lookat[1] = 9
        self.viewer.cam.lookat[2] = 5

        self.viewer.cam.elevation = -70  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)

class AntMazeEnv(MazeEnvV2):
    MODEL_CLASS = NewAntEnv

class AntManEnv(PacManEnv):
    def __init__(self,
                 cfg=None,
                 height=4, width=4,
                 reset_maze = True,
                 reset_goal = True,
                 block_size=20,
                 reward_type='sparse',
                 include_low_obs=1., penalty=0.2):
        Configurable.__init__(self)

        self.ant_env = AntMazeEnv(height, width, maze_size_scaling=4.8, wall_size=0.1)

        self.height = height
        self.width = width
        self.block_size = block_size
        self.reward_type = reward_type
        self.include_low_obs = include_low_obs
        self.penalty_weight = penalty

        self.observation_space = [
            Box(-np.inf, np.inf, (self.ant_env.observation_space.shape[0] + 4 + 2,)),
            Box(-np.inf, np.inf, (6 + (2 if include_low_obs else 0), height, width)),
        ][0]

        self.action_space = [Discrete(5), self.ant_env.action_space][1]

        self.reset_maze = reset_maze
        self.reset_goal = reset_goal
        self.maze = None
        import random
        state = random.getstate()
        random.seed(0)
        self.reset()
        random.setstate(state)


    def get_obs(self):
        high = self._high_background.copy()
        x, y = map(int, self.loc)
        high[4, y, x] = 1
        _x, _y = map(int, self.goal)
        high[5, _y, _x] = 1
        dx = self.loc - np.floor(self.loc)
        if self.include_low_obs > 0:
            high[6:8] = dx[:, None, None] * self.include_low_obs # should ignore it at the beginning..

        subgoal = tuple(map(int, self.subgoal))
        diff = np.array(subgoal) + 0.5 - self.loc

        obs = self.low_obs.copy()
        obs[:2] *= 0.0003
        #return [np.concatenate((obs, high[y, x][:4], diff)), high]

        return self.low_obs


    def reset(self):
        if self.reset_maze or self.maze is None:
            self.maze = MazeGame(self.height, width=self.width)
        self.maze.reset(reset_target=self.reset_goal)

        self._background = None
        self._high_background = get_maze_env_obs(self.maze, self.observation_space.shape[0])
        self.loc = np.array(self.maze.player) + 0.5
        self.goal = np.array(self.maze.target) + 0.5
        self.subgoal = tuple(map(int, self.loc))

        self.ant_env.wrapped_env.init_qpos[:2] = self.loc * self.ant_env.MAZE_SIZE_SCALING
        self.ant_env.set_map(self._high_background)
        self.low_obs = self.ant_env.reset()
        self.loc = self.low_obs[:2]/self.ant_env.MAZE_SIZE_SCALING
        return self.get_obs()

    def get_reward(self):
        loc = tuple(map(int, self.loc))
        goal = tuple(map(int, self.goal))
        subgoal = tuple(map(int, self.subgoal))
        high_reward = int(loc == goal) # - self.penalty * self.penalty_weight
        low_success = int(loc == subgoal)
        if self.reward_type == 'sparse':
            low_reward = int(loc == subgoal)
        else:
            low_reward = -np.linalg.norm(self.loc - (np.float32(subgoal) + 0.5))

        return {
            'meta': high_reward,
            'low': low_reward,
            'low_success': low_success,
            'success': int(loc == goal)
        }

    def step(self, action):
        self.low_obs, _, _, _ = self.ant_env.step(action)
        self.loc = self.low_obs[:2].copy()/self.ant_env.MAZE_SIZE_SCALING
        rewards = self.get_reward()
        low_success = rewards['low_success']
        del rewards['low_success']
        return self.get_obs(), rewards['meta'], False, {'success': rewards['success']}  

    def render(self, mode):
        if self._background is None and mode == 'rgb_array':
            self._background = render_background(self.maze, self.block_size)
        import cv2
        img1 = render_maze(self.maze, self.block_size, self._background, self.loc, self.goal, self.subgoal, mode)
        img2 = self.ant_env.render(mode='rgb_array')
        img1 = cv2.resize(img1[::-1], (img2.shape[1], img2.shape[0]))
        img = np.concatenate((img2, img1), axis=1)
        if mode == 'human':
            cv2.imshow('x', img)
            cv2.waitKey(1)
        else:
            return img

