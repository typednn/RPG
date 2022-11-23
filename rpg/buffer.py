import torch
import gym
import numpy as np
from tools.config import Configurable

class BufferItem:
    def __init__(self, capacity: int, space: dict) -> None:
        self._capacity = capacity
        self._space = space
        if '_shape' not in space:
            self._data = {k: BufferItem(capacity, v) for k, v in space.items()}
        else:
            self._data = torch.empty((capacity, *space['_shape']), dtype=space.get('_dtype', torch.float32), device=space.get('_device', 'cuda:0'))
            self._full = False
            self.idx = 0

    # def __setitem__(self, key, value):
    #     assert isinstance(key, slice)
    #     if isinstance(self._space, dict):
    #         for k, v in value.items():
    #             self._data[k][key] = v
    #     else:
    #         self._data[key] = value

    def __getitem__(self, key):
        if isinstance(self._data, dict):
            return {k: v[key] for k, v in self._data.items()}
        else:
            return self._data[key]

    def append(self, data):
        #idx = (idx + horizon) % capacity
        #    idx = min(idx + horizon, capacity)
        #return idx, full
        if isinstance(self._data, dict):
            for k, v in data.items():
                self._data[k].append(v)
        else:
            horizon = len(data)
            end = min(self.idx + horizon, self._capacity)
            self._data[self.idx:end] = data[:end-self.idx]

            new_idx = (self.idx + horizon) % self._capacity
            if self.idx + horizon >= self._capacity:
                self._full = True
                if new_idx != 0:
                    self._data[:new_idx] = data[end-self.idx:]
            self.idx = new_idx

    def __len__(self):
        if isinstance(self._data, dict):
            for k, v in self._data.items():
                return len(v)
        else:
            return self._capacity if self._full else self.idx

    @property
    def device(self):
        if isinstance(self._data, dict):
            for k, v in self._data.items():
                return v.device
        return self._data.device

    def sample_idx(self, batch_size):
        return torch.from_numpy(np.random.choice(len(self), batch_size)).to(self.device)


class ReplayBuffer(Configurable):
    # replay buffer with done ..
    def __init__(self, obs_space, action_space, episode_length, horizon,
                       cfg=None, device='cuda:0', max_episode_num=2000, modality='state', store_z=False):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.episode_length = episode_length
        self.capacity = max_episode_num * episode_length
        self.horizon = horizon

        assert modality == 'state'
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8

        if not isinstance(obs_space, dict):
            obs_shape = obs_space.shape
            self._obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device) # avoid last buggy..
            self._next_obs = torch.empty((self.capacity, *obs_shape), dtype=dtype, device=self.device)
            self.obs_shape = obs_shape
        else:
            self._obs = {}
            self._next_obs = {}
            for k, v in obs_space.items():
                self._obs[k] = torch.empty((self.capacity, *v.shape), dtype=dtype, device='cpu') # avoid last buggy..
                self._next_obs[k] = torch.empty((self.capacity, *v.shape), dtype=dtype, device='cpu')


        if isinstance(action_space, gym.spaces.Box):
            self._action = torch.empty((self.capacity, *action_space.shape), dtype=torch.float32, device=self.device)
        else:
            self._action = torch.empty((self.capacity,), dtype=torch.long, device=self.device)


        self._reward = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._truncated = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._timesteps = torch.empty((self.capacity,), dtype=torch.float32, device=self.device) - 1
        self._z = None

        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def total_size(self):
        if self._full:
            return self.capacity
        return self.idx

    @torch.no_grad()
    def add(self, traj):
        from .traj import Trajectory
        traj: Trajectory

        # assert self.episode_length == traj.timesteps, "episode length mismatch"
        length = traj.timesteps
        cur_obs = traj.get_tensor('obs', self.device)
        next_obs = traj.get_tensor('next_obs', self.device)
        actions = traj.get_tensor('a', self.device)
        rewards = traj.get_tensor('r', self.device)
        timesteps = traj.get_tensor('timestep', self.device)
        dones, truncated = traj.get_truncated_done(self.device)
        assert truncated[-1].all()

        if self._cfg.store_z:
            z = traj.get_tensor('z', self.device, dtype=None)
            if self._z is None:
                self._z = torch.empty((self.capacity, *z.shape[2:]), dtype=z.dtype, device=self.device)

        for i in range(traj.nenv):
            l = min(length, self.capacity - self.idx)


            if isinstance(self._obs, dict):
                for k, v in self._obs.items():
                    v[self.idx:self.idx+l] = cur_obs[k][:l, i]
                    self._next_obs[k][self.idx:self.idx+l] = next_obs[k][:l, i]
            else:
                self._obs[self.idx:self.idx+l] = cur_obs[:l, i]
                self._next_obs[self.idx:self.idx+l] = next_obs[:l, i]

            self._action[self.idx:self.idx+l] = actions[:l, i]
            self._reward[self.idx:self.idx+l] = rewards[:l, i]
            self._dones[self.idx:self.idx+l] = dones[:l, i, None]

            truncated[l-1, i] = True # last one is always truncated ..
            self._truncated[self.idx:self.idx+l] = truncated[:l, i, None]
            self._timesteps[self.idx:self.idx+l] = timesteps[:l, i]

            if self._z is not None:
                self._z[self.idx:self.idx+l] = z[:l, i]

            self.idx = (self.idx + l) % self.capacity
            self._full = self._full or self.idx == 0

    @torch.no_grad()
    def sample(self, batch_size, horizon=None):
        # NOTE that the data after truncated will be something random ..
        horizon = horizon or self.horizon
        total = self.total_size()
        idxs = torch.from_numpy(np.random.choice(total, batch_size, replace=not self._full)).to(self.device)

        if isinstance(self._obs, dict):
            obs_seq = []
        else:
            obs_seq = torch.empty((horizon + 1, batch_size, *self.obs_shape), dtype=torch.float32, device=self.device)

        timesteps = torch.empty((horizon + 1, batch_size), dtype=torch.float32, device=self.device)

        action = torch.empty((horizon, batch_size, *self._action.shape[1:]), dtype=self._action.dtype, device=self.device)
        reward = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        done = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        truncated = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)

        if isinstance(self._obs, dict):
            obs_seq.append({k: v[idxs.cpu()].to(self.device) for k, v in self._obs.items()})
        else:
            obs_seq[0] = self._obs[idxs]
        timesteps[0] = self._timesteps[idxs]

        if self._z is not None:
            z = self._z[idxs] # NOTE: we require the whole z to be the same ..


        for t in range(horizon):
            _idxs = (idxs + t).clamp(0, self.capacity-1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            done[t] = self._dones[_idxs]
            truncated[t] = self._truncated[_idxs]

            if isinstance(self._obs, dict):
                obs_seq.append({k: v[_idxs.cpu()].to(self.device) for k, v in self._next_obs.items()})
            else:
                obs_seq[t+1] = self._next_obs[_idxs]
            timesteps[t+1] = self._timesteps[_idxs] + 1


        truncated_mask = torch.ones_like(truncated) # we weill not predict state after done ..
        truncated_mask[1:] = 1 - (truncated.cumsum(0)[:-1] > 0).float()
        # raise NotImplementedError("The truncation seems incorrect in the end, need to be fixed")
        output = (obs_seq, timesteps, action, reward, done, truncated_mask[..., 0])
        if self._z is not None:
            output += (z,) #NOTE: this is the z before the state ..
        return output

    
    @torch.no_grad()
    def sample_start(self, batch_size):
        idx = torch.where(self._timesteps == 0)[0]
        select = torch.from_numpy(np.random.choice(len(idx), batch_size, replace=not self._full)).to(self.device)
        idx = idx[select]
        obs = self._obs[idx] if not isinstance(self._obs, dict) else {k: v[idx.cpu()].to(self.device) for k, v in self._obs.items()}
        return obs, self._z[idx], self._timesteps[idx]