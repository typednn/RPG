import torch
import numpy as np
from tools.config import Configurable
from .traj import Trajectory


class ReplayBuffer(Configurable):
    def __init__(self, obs_shape, action_dim, episode_length, horizon,
                       cfg=None, device='cuda:0', max_episode_num=2000,
                       modality='state', per_alpha=0.6, per_beta=0.4):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.episode_length = episode_length
        self.capacity = max_episode_num * episode_length
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.horizon = horizon

        assert modality == 'state'
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8

        self._obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device)
        self._action = torch.empty((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._last_obs = torch.empty((self.capacity//episode_length, *obs_shape), dtype=dtype, device=self.device)

        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def add(self, traj: Trajectory):
        assert self.episode_length == traj.timesteps, "episode length mismatch"
        cur_obs = traj.get_tensor('obs', self.device)
        next_obs = traj.get_tensor('next_obs', self.device)
        actions = traj.get_tensor('a', self.device)
        rewards = traj.get_tensor('r', self.device)

        for i in range(traj.nenv):
            self._obs[self.idx:self.idx+self.episode_length] = cur_obs[:, i]
            self._last_obs[self.idx//self.episode_length] = next_obs[-1, i]
            self._action[self.idx:self.idx+self.episode_length] = actions[:, i]
            self._reward[self.idx:self.idx+self.episode_length] = rewards[:, i]


            if self._full:
                max_priority = self._priorities.max().to(self.device).item()
            else:
                max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()

            mask = torch.arange(self.episode_length) >= self.episode_length-self.horizon
            new_priorities = torch.full((self.episode_length,), max_priority, device=self.device)
            new_priorities[mask] = 0
            self._priorities[self.idx:self.idx+self.episode_length] = new_priorities
            self.idx = (self.idx + self.episode_length) % self.capacity
            self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device).clamp(1e4) + self._eps

    def _get_obs(self, arr, idxs):
        if self.cfg.modality == 'state':
            return arr[idxs]
        raise NotImplementedError

    def sample(self, batch_size):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs = torch.empty((self.horizon+1, batch_size, *self.obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.horizon+1, batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.horizon+1, batch_size, 1), dtype=torch.float32, device=self.device)
        for t in range(self.horizon+1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs+1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs+1) % self.episode_length == 0
        next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.episode_length].cuda().float()
        # print(obs.shape, next_obs.shape, action.shape, reward.shape)

        return obs, next_obs, action, reward, idxs, weights