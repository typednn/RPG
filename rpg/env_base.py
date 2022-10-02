import numpy as np
from abc import ABC, abstractmethod
from tools.utils import tonumpy


class VecEnv(ABC):
    def __init__(self) -> None:
        self.nenv = None
        self.observation_space = None
        self.action_space = None

    @abstractmethod
    def start(self, **kwargs):
        # similar to reset, but return obs [batch of observations: nunmy array/dict/tensors] and timesteps [numpy].
        # if done or no obs, reset it.
        # note that this process is never differentiable.
        pass

    @abstractmethod
    def step(self, action):
        # return {'next_obs': _, 'obs': new_obs, 'reward': reward, 'done': done, 'info': info, 'total_reward': 0., 'truncated': False, 'episode': [], 'timestep'}
        # next obs will not be obs if done.
        pass

    def report(self, steps, returns):
        if not hasattr(self, 'episode_id'):
            self.episode_id = 0


class GymVecEnv(VecEnv):
    def __init__(self, env_name, n, ignore_truncated=True) -> None:
        import gym
        from rl.vec_envs import SubprocVectorEnv
        def make_env():
            return gym.make('HalfCheetah-v3')

        self.nenv = n
        self.vec_env = SubprocVectorEnv([make_env for i in range(n)])

        self.reset = False
        self.obs = None
        self.steps = None
        self.returns = None
        self.ignore_truncated = ignore_truncated

        self.observation_space = self.vec_env.observation_space[0]
        self.action_space = self.vec_env.action_space[0]

    def start(self, **kwargs):
        self.kwargs = kwargs

        if self.reset:
            pass
        else:
            self.obs = self.vec_env.reset(**kwargs) # reset all
            self.steps = np.zeros(len(self.obs), dtype=np.long)
            self.returns = np.zeros(len(self.obs), dtype=np.float)

        return self.obs, self.steps.copy()

    def step(self, actions):
        assert self.obs is not None, "must start before running"

        actions = tonumpy(actions)
        next_obs, reward, done, info = self.vec_env.step(actions)
        end_envs = np.where(done)[0]

        import copy
        obs = copy.copy(next_obs)

        self.steps += 1
        self.returns += np.array(reward)

        episode = []
        if len(end_envs) > 0:
            if self.ignore_truncated:
                for j in end_envs:
                    if 'TimeLimit.truncated' in info[j]:
                        done[j] = False

            for j in end_envs:
                step, total_reward = self.steps[j], self.returns[j]
                self.steps[j] = 0
                self.returns[j] = 0
                episode.append({'step': step, 'reward': total_reward})

            for idx, k in zip(end_envs, self.vec_env.reset(end_envs, **self.kwargs)):
                obs[idx] = k

        return {
            'obs': obs, # the current observation of the environment. 
            'next_obs': next_obs, # ending state of the previous transition.
            'timestep': self.steps.copy(),
            'r': np.array(reward)[:, None],
            'done': done,
            'info': info,
            'total_reward': self.returns.copy(),
            'truncated': [('TimeLimit.truncated' in i) for i in info],
            'episode': episode
        }