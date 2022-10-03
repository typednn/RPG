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
        assert vpred.shape == next_vpred.shape == reward.shape, "vpred and next_vpred must be the same length as reward"

        if debug:
            # sanity check below
            # compute GAE ..

            assert torch.allclose(
                traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale,
                traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=traj.nenv * traj.timesteps) * scale,
            )
            assert torch.allclose(
                traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale,
                traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=traj.nenv * traj.timesteps) * scale,
            )


            next_vpred_2 = torch.zeros_like(next_vpred)
            ind = traj.get_truncated_index()
            next_vpred_2[:-1] = vpred[1:]
            next_vpred_2 = traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size, index=ind, vpred=next_vpred_2) * scale
            assert torch.allclose(next_vpred, next_vpred_2)

            assert done.sum() == 0

        adv = torch.zeros_like(next_vpred)
        done = done.float()

        mask = (1-done)[..., None]
        assert mask.shape[:2] == next_vpred.shape[:2]

        lastgaelam = 0.
        for t in reversed(range(traj.timesteps)):
            m = mask[t]
            delta = reward[t] + self._cfg.gamma * next_vpred[t] * m - vpred[t]
            adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * m

        if debug:
            adv2 = compute_gae_by_hand(reward, vpred, next_vpred, done, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='approx')
            assert adv.shape == adv2.shape
            assert torch.allclose(adv, adv2)
            print(adv2[0])
            print(compute_gae_by_hand(reward, vpred, next_vpred, done, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='slow')[-8])
            print(compute_gae_by_hand(reward, vpred, next_vpred, done, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='exact')[-8])

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


def compute_gae_by_hand(reward, value, next_value, done, gamma, lmbda, mode='approx'):
    reward = reward.to(torch.float64)
    value = value.to(torch.float64)
    next_value = next_value.to(torch.float64)
    # follow https://arxiv.org/pdf/1506.02438.pdf
    if mode != 'exact':
        import tqdm
        gae = []
        for i in tqdm.trange(len(reward)):
            adv = 0.
            # legacy ..

            if mode == 'approx':
                for j in range(i, len(reward)):
                    delta = reward[j] + next_value[j] * gamma - value[j]
                    adv += (gamma * lmbda)**(j-i) * delta * (1. - done[j].float())[..., None]
            elif mode == 'slow':
                R = 0
                discount_gamma = 1.
                discount_lmbda = 1.
                for j in range(i, len(reward)):
                    mask = (1. - done[j].float())[..., None]
                    discount_gamma = discount_gamma * mask

                    R = R + reward[j] * discount_gamma
                    A = R + (discount_gamma * gamma) * next_value[j] - value[i]

                    discount_lmbda = discount_lmbda * mask
                    adv += (A * discount_lmbda) * (1-lmbda)

                    discount_gamma = discount_gamma * gamma
                    discount_lmbda = discount_lmbda * lmbda
            else:
                raise NotImplementedError

            gae.append(adv)
    else:
        """
        1               -V(s_t)  + r_t                                                                                     + gamma * V(s_{t+1})
        lmabda          -V(s_t)  + r_t + gamma * r_{t+1}                                                                   + gamma^2 * V(s_{t+2})
        lambda^2        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                                               + ...
        lambda^3        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
        """
        sum_lambda = 0.
        sum_reward = 0.
        sum_end_v = 0.
        gae = []
        mask = (1. - done.float())[..., None]

        for i in reversed(range(len(reward))):
            sum_lambda = sum_lambda * mask[i]
            sum_reward = sum_reward * mask[i]
            sum_end_v = sum_end_v * mask[i]

            sum_lambda = 1. + lmbda * sum_lambda
            sum_reward = lmbda * gamma * sum_reward + sum_lambda * reward[i]
            sum_end_v = sum_end_v * gamma + next_value[i]  * gamma
            sumA = sum_reward + sum_end_v - value[i] * sum_lambda
            gae.append(sumA * (1-lmbda))
        gae = gae[::-1]

    return torch.stack(gae).float() #* (1-lmbda) 


class HierarchicalGAE(Configurable):
    # @ litian, write the unit test code for the target value computation..
    def __init__(self, pi_a, pi_z, cfg=None, gamma=0.995, lmbda=0.97, adv_norm=True):
        super().__init__()
        self.pi_a = pi_a
        self.pi_z = pi_z


    @torch.no_grad()
    def __call__(self, traj: Trajectory, reward: torch.Tensor, batch_size: int, rew_rms):
        done = traj.get_tensor('done')

        scale = rew_rms.std if rew_rms is not None else 1.
        vpredz = traj.predict_value(
            ('obs', 'old_z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale
        vpreda = traj.predict_value(
            ('obs', 'z', 'timestep'), self.pi_a.value, batch_size=batch_size) * scale
        next_vpredz = traj.predict_value(
            ('next_obs', 'z', 'timestep'), self.pi_z.value, batch_size=batch_size) * scale

        next_vpreda = torch.zeros_like(next_vpredz)
        ind = traj.get_truncated_index()
        next_vpreda[:-1] = vpreda[1:]
        # there is no action estimate as we did not predict z 
        next_vpreda[ind[:, 0], ind[:, 1]] = next_vpredz[ind[:, 0], ind[:, 1]] 

        # compute GAE ..
        lmbda_sqrt = self._cfg.lmbda**0.5

        vpred = vpredz + vpreda * lmbda_sqrt
        next_vpred = vpredz * vpreda * lmbda_sqrt
        adv = torch.zeros_like(next_vpred)
        lastgaelam = 0.
        for t in reversed(range(traj.timesteps)):
            nextvalue = next_vpred[t]
            mask = (1. - done[t].float())[..., None]
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
