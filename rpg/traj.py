import numpy as np
import torch
from typing import List, Dict, Any
from .utils import iter_batch
from tools.utils import totensor
from .utils import minibatch_gen

class DataBuffer(dict):
    def loop_over(self, batch_size, keys=None):
        # TODO: preserve the trajectories if necessay.
        if keys is not None:
            return DataBuffer(**{key: self[key] for key in keys}).loop_over(batch_size)

        timesteps = len(self['obs'])
        nenvs = len(self['obs'][0])

        import numpy as np
        index = np.stack(np.meshgrid(np.arange(timesteps), np.arange(nenvs)), axis=-1).reshape(-1, 2)
        return minibatch_gen(self, index, batch_size)


# dataset 
class Trajectory:
    def __init__(self, transitions: List[Dict], nenv, timesteps) -> None:
        self.traj = transitions
        self.nenv = nenv
        self.timesteps = timesteps
        assert len(self.traj) == self.timesteps

        self.index = np.array([(i, j) for j in range(nenv) for i in range(timesteps)])

    def __add__(self, other: "Trajectory"):
        assert self.nenv == other.nenv
        return Trajectory(self.traj + other.traj, self.nenv, self.timesteps + other.timesteps)

    def predict_value(self, key, network, batch_size, index=None, vpred=None):
        if isinstance(key, str):
            key = [key]

        if index is None:
            index = self.index

        for ind in iter_batch(index, batch_size):
            obs = [[self.traj[i][k][j] if self.traj[i][k] is not None else None for i, j in ind] for k in key]
            value = network(*obs)

            if vpred is None:
                vpred = torch.zeros((self.timesteps, self.nenv,
                    *value.shape[1:]), device=value.device, dtype=value.dtype)

            ind = totensor(ind, dtype=torch.long, device=value.device)
            vpred[ind[:, 0], ind[:, 1]] = value
        return vpred

    def get_tensor(self, key, device='cuda:0'):
        from tools.utils import totensor, dstack
        return totensor([i[key] for i in self.traj], device=device)

    def get_list_by_keys(self, keys) -> DataBuffer:
        return DataBuffer(**{key: [i[key] for i in self.traj] for key in keys})

    def get_truncated_index(self) -> np.ndarray:
        ind = []
        for j in range(self.timesteps):
            for i in range(self.nenv):
                if self.traj[j]['truncated'][i] or j == self.timesteps -1:
                    ind.append((j, i))
        return np.array(ind)
        #return totensor(ind, dtype=torch.long, device=device)


    def summarize_epsidoe_info(self):
        # average additional infos, for example success, if necessary.
        n = 0
        rewards = 0
        avg_len = 0
        for i in range(self.timesteps):
            if 'episode' in self.traj[i]:
                for j in self.traj[i]['episode']:
                    n += 1
                    rewards += j['reward']
                    avg_len += j['step']
        return {
            "num_episode": n,
            "rewards": rewards / n,
            "avg_len": avg_len / n,
        }