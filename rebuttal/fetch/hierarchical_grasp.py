import numpy as np
import gym
from .sequential import SequentialStack, register
from ..message import Message


class HighPolicy:
    def __init__(self, env):
        self.n_step = 0

    def reset(self, state, mode='train'):
        self.n_step = 0
        return 0

    def select_action(self, trans):
        if trans[2]:
            return None
        self.n_step += 1
        return self.n_step

    def save(self, *args, **kwargs):
        pass

class HierarchicalGrasp(SequentialStack):
    def __init__(self, cfg=None, double_action=False, include_stage_id=False):
        super(HierarchicalGrasp, self).__init__(cfg=cfg)
        assert not double_action

        self.include_stage_id = include_stage_id
        high_obs = gym.spaces.Box(-np.inf, np.inf, (self.observation_space.shape[0] + (self.include_stage_id
                                                                                        - self.stage_dim) * self.num_blocks,))
        high_obs.shared_dim = 10 + int(self.include_step_in_obs)
        high_obs.num_blocks = self.num_blocks
        high_obs.goal_dim = 3
        high_obs.object_dim = 15 + int(self.include_stage_id)

        self.observation_space = [self.observation_space, high_obs]
        self.action_space = [self.action_space, gym.spaces.Discrete(n=self.num_blocks)]
        self._loop = None
        self._substeps = self.substeps
        self._max_episode_steps = self.num_blocks * self._substeps
        self.high_agent = HighPolicy(self)
        self.include_stage_id = include_stage_id

    def step_meta(self, action):
        self.obj_id = action
        self.stage_id = 0
        self.stage_step = 0
        obs = self._get_obs()
        return [obs, obs]

    def step(self, action):
        action = np.array(action).clip(-1, 1)

        self.compute_prev_reward()
        self.obs, r, _, info = self.env.step(action)
        reward = self.compute_reward_info(info)

        if self.double_stage:
            self.stage_step += 1
            if self.stage_step % self.double_stage_steps == 0:
                self.stage_id = 1

        self.steps += 1

        obs = self._get_obs()
        done = False
        return [obs, self._get_obs(include_stage=False, include_stage_id=self.include_stage_id)], [reward, r], done, info

    def reset(self):
        obs = super(HierarchicalGrasp, self).reset()
        return [obs, self._get_obs(include_stage=False, include_stage_id=self.include_stage_id)]


register(HierarchicalGrasp)