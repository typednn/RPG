# ant 
import gym
from envs.pacman.antman import AntMazeEnv
import numpy as np


class TripleAntEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        width = 4
        height = 4
        self.ant_env = AntMazeEnv(4, 4, maze_size_scaling=4.8, wall_size=0.1, lookat=(0, 0, 0))


        self.observation_space = self.ant_env.observation_space
        self.action_space = self.ant_env.action_space


        self._high_background = np.zeros((4, height+1, width+1))
        self.ant_env.set_map(self._high_background)
        self.loc = np.zeros(2)

    def get_obs(self):
        return self.low_obs.copy()

    def reset(self):
        self.ant_env.wrapped_env.init_qpos[:2] = 0. #self.loc * self.ant_env.MAZE_SIZE_SCALING
        self.low_obs = self.ant_env.reset()
        return self.get_obs()

    def step(self, action):
        self.low_obs, _, _, _ = self.ant_env.step(action)

        self.loc = self.low_obs[:2].copy() #/self.ant_env.MAZE_SIZE_SCALING

        goals = np.array(
            [
                [0                      , -1.], 
                [1./2 * 3 ** 0.5 , 1./2],
                [-1./2 * 3 ** 0.5, 1./2]
            ], 
        ) * self.ant_env.MAZE_SIZE_SCALING
        reward = -np.linalg.norm( (self.loc[None, :2] - goals[:, :2]), axis=-1)
        reward = reward.min(axis=-1)
        return self.get_obs(), reward, False, {}

    def render(self, mode='rgb_array'):
        return self.ant_env.render(mode=mode)