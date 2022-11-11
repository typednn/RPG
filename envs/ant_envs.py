# base environment for hierarhical RL

import gym
import numpy as np
from gym.spaces import Dict, Box
# this replies on hiro_robot_envs...
# TODO: move envs into my repos.
from .hiro_robot_envs.create_maze_env import create_maze_env

HIGH = 'meta'
LOW = 'low'


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training.
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    """
    get the reward function.
    Arguments
    ---------
    `env_name`: `str`: The possible environment names. Each reward function
    is the negative distance to the goal.
    Possible environment names:
    - `'AntMaze'`
    - `'AntPush'`
    - `'AntFall'`
    """
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


class AntHEnv(gym.Env):
    # remember:
    #   1. action rescale to -1, 1
    #   2. provide relabel func
    #   3. provide high, low observations
    #   4. provide extra states [real goal] in states

    def __init__(self, env_name, env_subgoal_dim=15, goal_threshold=5):
        self.base_env = create_maze_env(env_name, include_block_in_obs=False)
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 5
        self.goal_threshold = goal_threshold
        self.metadata = {'render.modes': ['human', 'rgb_array', 'depth_array'], 'video.frames_per_second': 125}
        # create general subgoal space, independent of the env
        self.subgoal_dim = env_subgoal_dim
        limits = np.array([10, 10, 0.5, 1, 1, 1, 1,
                           0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3])[:self.subgoal_dim]

        self.subgoal_space = gym.spaces.Box(low=-limits, high=limits)

        if env_name == 'AntMaze' or env_name == 'AntPush':
            env_goal_dim = 2
        else:
            env_goal_dim = 3
        self.goal_dim = env_goal_dim

        self.substeps = 10
        self.observation_space = Dict(
            low=Box(-np.inf, np.inf, (self.base_env.observation_space.shape[0] + self.subgoal_dim,)),
            meta=Box(-np.inf, np.inf, (self.base_env.observation_space.shape[0] + self.goal_dim,)),
        )['meta']
        self.action_space = Dict(
            low=self.base_env.action_space,
            meta=self.subgoal_space,
        )['low']
        self._subgoal = self.subgoal_space.sample()
        self.default_substeps = 10
        self._max_episode_steps = 500

    def _get_obs(self):
        # we first follow the previous implementation..
        low = np.r_[
            self._base_obs, self._subgoal - self._base_obs[:self.subgoal_dim]].copy()  # only input the relative goal ..
        low[:2] = 0  # # Zero-out x, y position. Hacky. WTF!!!!!!!!!
        return {
            HIGH: np.r_[self._base_obs, self.goal],  # input the original goal..
            LOW: low
        }[HIGH]

    def _get_reward(self):
        meta_reward = self.reward_fn(self._base_obs, self.goal)
        return {
            HIGH: meta_reward,
            LOW: -np.linalg.norm(self._subgoal - self._base_obs[:self.subgoal_dim])
        }[HIGH]

    def seed(self, seed=None):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        self._base_obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        self._subgoal = self._base_obs[:self.subgoal_dim]
        return self._get_obs()

    # def step_meta(self, subgoal):
    #     self._subgoal = self._base_obs[:self.subgoal_dim] + subgoal
    #     return self._get_obs()

    def step(self, a):
        self._base_obs, _, done, info = self.base_env.step(a)
        r = self._get_reward()
        return self._get_obs(), r, done, {'success': r > -self.goal_threshold}

    def render(self, mode='rgb_array'):
        img = self.base_env.render(mode='rgb_array')
        if mode == 'human':
            img = img[..., ::-1]
            import cv2
            cv2.imshow('x', img)
            cv2.waitKey(1)
            return None
        else:
            return img