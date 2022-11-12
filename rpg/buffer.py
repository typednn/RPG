import torch
import gym
import numpy as np
from tools.config import Configurable
from .traj import Trajectory


class ReplayBuffer(Configurable):
    # replay buffer with done ..
    def __init__(self, obs_shape, action_space, episode_length, horizon,
                       cfg=None, device='cuda:0', max_episode_num=2000, modality='state'):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.episode_length = episode_length
        self.capacity = max_episode_num * episode_length
        self.obs_shape = obs_shape
        self.horizon = horizon

        assert modality == 'state'
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8

        self._obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device) # avoid last buggy..
        self._next_obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device)

        if isinstance(action_space, gym.spaces.Box):
            self._action = torch.empty((self.capacity, *action_space.shape), dtype=torch.float32, device=self.device)
        else:
            self._action = torch.empty((self.capacity,), dtype=torch.long, device=self.device)


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

            truncated[l-1, i] = True # last one is always truncated ..
            self._truncated[self.idx:self.idx+l] = truncated[:l, i, None]
            self._timesteps[self.idx:self.idx+l] = timesteps[:l, i]

            self.idx = (self.idx + l) % self.capacity
            self._full = self._full or self.idx == 0

    @torch.no_grad()
    def sample(self, batch_size):
        # NOTE that the data after truncated will be something random ..
        total = self.total_size()
        idxs = torch.from_numpy(np.random.choice(total, batch_size, replace=not self._full)).to(self.device)

        obs_seq = torch.empty((self.horizon + 1, batch_size, *self.obs_shape), dtype=torch.float32, device=self.device)
        timesteps = torch.empty((self.horizon + 1, batch_size), dtype=torch.float32, device=self.device)

        action = torch.empty((self.horizon, batch_size, *self._action.shape[1:]), dtype=self._action.dtype, device=self.device)
        reward = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        done = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        truncated = torch.empty((self.horizon, batch_size, 1), dtype=torch.float32, device=self.device)

        _done = None
        obs_seq[0] = self._obs[idxs]
        timesteps[0] = self._timesteps[idxs]

        for t in range(self.horizon):
            _idxs = (idxs + t).clamp(0, self.capacity-1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            done[t] = self._dones[_idxs]
            truncated[t] = self._truncated[_idxs]

            obs_seq[t+1] = self._next_obs[_idxs]
            timesteps[t+1] = self._timesteps[_idxs] + 1


        truncated_mask = torch.ones_like(truncated) # we weill not predict state after done ..
        truncated_mask[1:] = 1 - (truncated.cumsum(0)[:-1] > 0).float()
        # raise NotImplementedError("The truncation seems incorrect in the end, need to be fixed")
        return obs_seq, timesteps, action, reward, done, truncated_mask[..., 0]