import torch
from tools.utils import totensor
from tools.utils import dstack, totensor
from .traj import Trajectory
from .ppo_agent import PPOAgent
from tools.config import Configurable


class GAE(Configurable):
    def __init__(self, pi: PPOAgent, cfg=None, gamma=0.995, lmbda=0.97, adv_norm=True):
        super().__init__()
        self.pi = pi

    @torch.no_grad()
    def __call__(self, traj: Trajectory, reward: torch.Tensor, batch_size: int, rew_rms, debug=False):
        done = traj.get_tensor('done')

        scale = rew_rms.std if rew_rms is not None else 1.

        vpred = traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale
        next_vpred = traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale

        if debug:
            # sanity check below
            assert torch.allclose(
                traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale,
                traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=traj.nenv * traj.timesteps) * scale,
            )
            assert torch.allclose(
                traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale,
                traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=traj.nenv * traj.timesteps) * scale,
            )

        assert vpred.shape == next_vpred.shape == reward.shape, "vpred and next_vpred must be the same length as reward"

        if debug:
            next_vpred_2 = torch.zeros_like(next_vpred)
            ind = traj.get_truncated_index()
            next_vpred_2[:-1] = vpred[1:]
            next_vpred_2 = traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size, index=ind, vpred=next_vpred_2) * scale
            assert torch.allclose(next_vpred, next_vpred_2)

        # compute GAE ..
        if debug:
            assert done.sum() == 0

        adv = torch.zeros_like(next_vpred)

        mask = (1-done)[..., None]
        assert mask.shape[:2] == next_vpred.shape[:2]

        lastgaelam = 0.
        for t in reversed(range(traj.timesteps)):
            m = mask[t]
            delta = reward[t] + self._cfg.gamma * next_vpred[t] * m - vpred[t]
            adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * m

        vtarg = vpred + adv

        if rew_rms is not None:
            # https://github.com/haosulab/pyrl/blob/926d3d07d45f3bf014e7c6ea64e1bba1d4f35f03/pyrl/utils/torch/module_utils.py#L192
            rew_rms.update(vtarg.sum(axis=-1).reshape(-1)) # normalize it based on the final result.
            scale = rew_rms.std
        else:
            scale = 1.

        adv = (vtarg-vpred)/scale
        vtarg = vtarg/scale

        if self._cfg.adv_norm:
            adv = adv - adv.mean()
            adv = adv/(adv.std() + 1e-9)

        return dict(
            adv=adv,
            vtarg = vtarg,
        )


class HierarchicalGAE(Configurable):
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