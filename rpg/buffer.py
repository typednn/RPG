import torch
import numpy as np
from tools.config import Configurable
from .traj import Trajectory


class ReplayBuffer(Configurable):
    # replay buffer with done ..
    def __init__(self, obs_shape, action_dim, episode_length, horizon,
                       cfg=None, device='cuda:0', max_episode_num=2000, modality='state'):
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

        self._obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device) # avoid last buggy..
        self._next_obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device)

        self._action = torch.empty((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._truncated = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._timesteps = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)

        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def total_size(self):
        if self._full:
            return self.capacity
        return self.idx

    @torch.no_grad()
    def add(self, traj: Trajectory):
        # assert self.episode_length == traj.timesteps, "episode length mismatch"
        length = traj.timesteps
        cur_obs = traj.get_tensor('obs', self.device)
        next_obs = traj.get_tensor('next_obs', self.device)
        actions = traj.get_tensor('a', self.device)
        rewards = traj.get_tensor('r', self.device)
        timesteps = traj.get_tensor('timestep', self.device)
        dones, truncated = traj.get_truncated_done(self.device)
        assert truncated[-1].all()

        for i in range(traj.nenv):
            l = min(length, self.capacity - self.idx)


            self._obs[self.idx:self.idx+l] = cur_obs[:l, i]
            self._next_obs[self.idx:self.idx+l] = next_obs[:l, i]
            self._action[self.idx:self.idx+l] = actions[:l, i]
            self._reward[self.idx:self.idx+l] = rewards[:l, i]
            self._dones[self.idx:self.idx+l] = dones[:l, i, None]
            self._truncated[self.idx:self.idx+l] = truncated[:l, i, None]
            self._timesteps[self.idx:self.idx+l] = timesteps[:l, i]

            self.idx = (self.idx + l) % self.capacity
            self._full = self._full or self.idx == 0

    @torch.no_grad()
    def sample(self, batch_size):
        # NOTE that the data after truncated will be something random ..
        total = self.total_size()
        idxs = torch.from_numpy(np.random.choice(total, batch_size, replace=not self._full)).to(self.device)

        #obs = self._get_obs(self._obs, idxs)
        obs = self._obs[idxs]
        next_obs = torch.empty((self.horizon, batch_size, *self.obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.horizon, batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        done = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        truncated = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        timesteps = torch.empty((self.horizon, batch_size), dtype=torch.float32, device=self.device)

        _done = None
        for t in range(self.horizon):
            _idxs = (idxs + t).clamp(0, self.capacity-1)
            action[t] = self._action[_idxs]
            next_obs[t] = self._next_obs[_idxs]
            reward[t] = self._reward[_idxs]
            if _done is None:
                _done = self._dones[_idxs]
            else:
                _done = torch.maximum(_done, self._dones[_idxs]) # once done, forever done
            done[t] = _done
            truncated[t] = self._truncated[_idxs]
            timesteps[t] = self._timesteps[_idxs]

        return obs, next_obs, action, reward, done, truncated, timesteps