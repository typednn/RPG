import numpy as np
from gym.spaces import Box
import gym

class OneLayerWrapper(gym.Env):
    """
    hiro like two layer env.
    """
    def __init__(self, env, max_timesteps):
        super(OneLayerWrapper, self).__init__()
        self.env = env
        shape = self.env.observation_space['meta'].shape
        self.observation_space = Box(-np.inf, np.inf, (8,) + shape[1:])
        self.action_space = self.env.action_space['low']
        self._max_episode_steps = max_timesteps

    def merge_obs(self, obs):
        a = self.env.reset()['meta']
        b = self.env.reset()['low']
        a = np.concatenate((a, a[:2] * 0 + b[:2, None, None]))
        return a

    def merge_reward(self, reward):
        return reward['meta']

    def reset(self):
        self.steps = 0
        return self.merge_obs(self.env.reset())

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self._max_episode_steps:
            done = True
        return self.merge_obs(obs), self.merge_reward(r), done, info

    def render(self, mode='human'):
        return self.env.render(mode)
