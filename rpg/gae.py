import torch
from tools.utils import totensor
from tools.utils import dstack, totensor
from .traj import Trajectory
from tools.config import Configurable


class HierarchicalGAE(Configurable):
    # @ litian, write the unit test code for the target value computation..
    def __init__(self, pi_a, pi_z, cfg=None, gamma=0.995, lmbda=0.97, adv_norm=True):
        super().__init__()
        self.pi_a = pi_a
        self.pi_z = pi_z


    @torch.no_grad()
    def __call__(self, traj: Trajectory, reward: torch.Tensor, batch_size: int, rew_rms):
        done, truncated = traj.get_truncated_done()

        scale = rew_rms.std if rew_rms is not None else 1.
        vpredz = traj.predict_value(
            ('obs', 'old_z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale
        vpreda = traj.predict_value(
            ('obs', 'z', 'timestep'), self.pi_a.value, batch_size=batch_size) * scale
        next_vpredz = traj.predict_value(
            ('next_obs', 'z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale

        next_vpreda = torch.zeros_like(next_vpredz)
        ind = traj.get_truncated_index(include_done=True) # because for the done, we still do not have next z to get the next_a prediction.
        next_vpreda[:-1] = vpreda[1:]
        next_vpreda[ind[:, 0], ind[:, 1]] = next_vpredz[ind[:, 0], ind[:, 1]] 
        assert len(ind) == truncated.sum()

        # compute GAE ..
        lmbda_sqrt = self._cfg.lmbda**0.5
        done = done.float()
        truncated = truncated.float()

        next_vpred = (next_vpredz + next_vpreda * lmbda_sqrt)/(1 + lmbda_sqrt)
        total, sum_weights, last_values = compute_gae_by_hand(reward, None, next_vpred, done, truncated, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='exact', return_sum_weight_value=True)

        def weighted_sum(values, last, weights, total_weight):
            total = 0
            weight = 0
            for a, b in zip(values, weights):
                total = total + a
                weight = weight + b
            total = total + (total_weight - weight) * last
            return total/total_weight

        vtarg = weighted_sum((total,), last_values, (sum_weights,), 1./ (1.-self._cfg.lmbda)).float()
        vtarg_z = weighted_sum((vpreda, total * lmbda_sqrt), last_values, (1., sum_weights * lmbda_sqrt), 1./ (1.-self._cfg.lmbda))


        if rew_rms is not None:
            # https://github.com/haosulab/pyrl/blob/926d3d07d45f3bf014e7c6ea64e1bba1d4f35f03/pyrl/utils/torch/module_utils.py#L192
            rew_rms.update(vtarg.sum(axis=-1).reshape(-1)) # normalize it based on the final result.
            scale = rew_rms.std
        else:
            scale = 1.

        adv_a = (vtarg - vpreda) / scale
        vtarg_a = vtarg / scale
        adv_z = (vtarg_z - vpredz) / scale
        vtarg_z = vtarg_z / scale

        if self._cfg.adv_norm:
            adv_a = adv_a - adv_a.mean()
            adv_a = adv_a/(adv_a.std() + 1e-9)

            adv_z = adv_z - adv_z.mean()
            adv_z = adv_z/(adv_z.std() + 1e-9)


        return dict(
            adv_a=adv_a,
            adv_z=adv_z,
            vtarg_a=vtarg_a,
            vtarg_z=vtarg_z,
        )