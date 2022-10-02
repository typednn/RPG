import torch
from tools.utils import totensor
from tools.utils import dstack, totensor
from .traj import Trajectory
from tools.config import Configurable



class ComputeTargetValues(Configurable):
    # @ litian, write the unit test code for the target value computation..
    def __init__(self, pi_a, pi_z, cfg=None, gamma=0.995, lmbda=0.97):
        super().__init__()
        self.pi_a = pi_a
        self.pi_z = pi_z


    @torch.no_grad()
    def __call__(self, traj: Trajectory, reward: torch.Tensor, batch_size: int, rew_rms):
        done = traj.get_tensor(traj, 'done')

        scale = rew_rms.std if rew_rms is not None else 1.
        vpredz = traj.predict_value(
            traj, ('obs', 'old_z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale
        vpreda = traj.predict_value(
            traj, ('obs', 'z', 'timestep'), self.pi_a.value, batch_size=batch_size) * scale
        next_vpredz = traj.predict_value(
            traj, ('next_obs', 'z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale

        next_vpreda = torch.zeros_like(next_vpredz)
        ind = traj.get_temrinal_inds()
        next_vpred[:-1] = vpreda[1:]
        next_vpreda[ind] = next_vpredz[ind] # there is no action estimate .. 

        # compute GAE ..
        lmbda_sqrt = self._cfg.lmbda**0.5

        vpred = vpredz + vpreda * lmbda_sqrt
        next_vpred = vpredz * vpreda * lmbda_sqrt
        adv = torch.zeros_like(next_vpred)
        for t in reversed(len(vpredz)):
            nextvalue = next_vpred[t]
            mask = 1. - done[t]
            assert mask.shape == nextvalue.shape
            #print(reward.device, next_vpred.device, mask.device, vpred[t].device)
            delta = reward[t] + self._cfg.gamma * nextvalue * mask - vpred[t]
            adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * mask
        vtarg = vpred + adv

        if rew_rms is not None:
            # https://github.com/haosulab/pyrl/blob/926d3d07d45f3bf014e7c6ea64e1bba1d4f35f03/pyrl/utils/torch/module_utils.py#L192
            rew_rms.update(vtarg.sum(axis=-1).reshape(-1)) # normalize it based on the final result.
            scale = rew_rms.std
        else:
            scale = 1.

        vtarg_z = vpreda + lmbda_sqrt * vtarg 
        return dict(
            adv_a = (vtarg - vpreda) / scale,
            vtarg_a = vtarg / scale,
            adv_z = (vtarg_z - vpredz) / scale,
            vtarg_z = vtarg_z / scale,
        )