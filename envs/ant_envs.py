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

    def __init__(self, env_name, env_subgoal_dim=15, obs_dim=6, reward_type='dense', goal_threshold=5):
        self.base_env = create_maze_env(env_name, include_block_in_obs=False)
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 5
        self.goal_threshold = goal_threshold
        self.reward_type = reward_type
        self.metadata = {'render.modes': ['human', 'rgb_array', 'depth_array'], 'video.frames_per_second': 125}
        # create general subgoal space, independent of the env
        self.subgoal_dim = env_subgoal_dim
        limits = np.array([10, 10, 0.5, 1, 1, 1, 1,
                           0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3])[:self.subgoal_dim]

        self.subgoal_space = gym.spaces.Box(low=-limits, high=limits)
        from . import utils

        if env_name == 'AntMaze' or env_name == 'AntPush':
            env_goal_dim = 2
        else:
            env_goal_dim = 3
        self.embedder, dim = utils.get_embeder_np(obs_dim, env_goal_dim)

        self.goal_dim = env_goal_dim

        self.substeps = 10
        self.action_space = self.base_env.action_space
        self._subgoal = self.subgoal_space.sample()
        self.default_substeps = 10
        self._max_episode_steps = 500

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.reset().shape)

    def _get_obs(self):
        inp = self.embedder(self._base_obs[:self.subgoal_dim]/20.)
        if self.env_name == 'AntPush':
            #print()
            movable = self.base_env.wrapped_env.get_body_com('moveable_2_2')
            inp = np.concatenate((inp, self.embedder(movable[:2] / 20.)))
        return np.r_[self._base_obs * 0.025, self.goal * 0.025, inp]  # input the original goal..

    def seed(self, seed=None):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        self._base_obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        self._subgoal = self._base_obs[:self.subgoal_dim]
        return self._get_obs()

    def step(self, a):
        self._base_obs, _, done, info = self.base_env.step(a)
        negative_dist = self.reward_fn(self._base_obs, self.goal)
        #r = self._get_reward()
        reward = negative_dist
        
        if self.reward_type == 'sparse':
            reward = float(negative_dist > -self.goal_threshold)
        return self._get_obs(), reward, done, {'success': negative_dist > -self.goal_threshold}

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

    def _render_traj_rgb(self, traj, z=None, occ_val=False, verbose=True, history=None, **kwargs):
        from .utils import extract_obs_from_tarj
        obs = (extract_obs_from_tarj(traj)/0.025)[..., :2]
        images = {}
        output = {
            'state': obs,
            'background': {
                'image':  None,
            },
            'history': {},
            'image': images,
            'metric': {},
        }

        return output

if __name__ == '__main__':
    env = AntHEnv('AntPush')
    env.reset()
    images = []
    for i in range(100):
        env.step(env.action_space.sample())
        images.append(env.render('rgb_array'))
    env.close()

    from tools.utils import animate
    animate(images)