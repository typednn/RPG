import numpy as np
import torch
from typing import List, Dict, Any
from .utils import sample_batch
from tools.utils import totensor


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

    def predict_value(self, key, index, network, batch_size, vpred=None):
        if isinstance(key, str):
            key = [key]

        index = index or self.index

        for ind in sample_batch(index, batch_size):
            obs = [[self.traj[i][k][j] for i, j in ind] for k in key]
            value = network(*obs)
            if vpred is None:
                vpred = torch.zeros((self.timesteps, self.nenv,
                    *value.shape[1:]), device=value.device, dtype=value.dtype)
            ind = torch.tensor(ind, dtype=torch.long, device=value.device)
            vpred[ind[0], ind[1]] = value
        return vpred

    def get_tensor(self, key):
        from tools.utils import totensor, dstack
        return dstack([totensor(i[key]) for i in self.traj])

    def get_list(self, key):
        return [i[key] for i in self.traj]

    def get_temrinal_inds(self):
        ind = []
        for j in range(self.timesteps):
            for i in range(self.nenv):
                if self.traj[j]['done'][i] or j == self.timesteps -1:
                    ind.append(j, i)
        return totensor(ind, device=torch.long)


    def summarize_epsidoe_info(self):
        # average additional infos, for example success, if necessary.
        n = 0
        rewards = 0
        avg_len = 0
        for i in range(len(self.timesteps)):
            if 'episode' in i:
                for j in i:
                    n += 1
                    rewards += j['reward']
                    avg_len += j['step']
        return {
            "num_episode": n,
            "rewards": rewards / n,
            "avg_len": avg_len / n,
        }