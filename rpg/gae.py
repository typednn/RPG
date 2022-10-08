import torch
from tools.utils import totensor
from tools.utils import dstack, totensor
from .traj import Trajectory
from .ppo_agent import PPOAgent
from tools.config import Configurable


class GAE(Configurable):
    def __init__(self, pi: PPOAgent, cfg=None, gamma=0.995, lmbda=0.97, adv_norm=True, correct_gae=False, ignore_done=True):
        super().__init__()
        self.pi = pi

    @torch.no_grad()
    def __call__(self, traj: Trajectory, reward: torch.Tensor, batch_size: int, rew_rms, debug=False):
        done, truncated = traj.get_truncated_done()
        #done = traj.get_tensor('done')
        if self._cfg.ignore_done:
            done = done * 0

        scale = rew_rms.std if rew_rms is not None else 1.

        vpred = traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale
        next_vpred = traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=batch_size) * scale
        assert vpred.shape == next_vpred.shape == reward.shape, "vpred and next_vpred must be the same length as reward"

        #done = done.float()
        if not self._cfg.correct_gae:
            adv = torch.zeros_like(next_vpred)

            mask = (1-truncated.float())[..., None]
            assert mask.shape[:2] == next_vpred.shape[:2]

            lastgaelam = 0.
            for t in reversed(range(traj.timesteps)):
                m = mask[t]
                delta = reward[t] + self._cfg.gamma * next_vpred[t] * (1-done[t].float())[..., None] - vpred[t] #TODO: modify it to truncated later.
                adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * m
        else:
            adv = compute_gae_by_hand(reward, vpred, next_vpred, done, truncated, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='exact')

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


def compute_gae_by_hand(reward, value, next_value, done, truncated, gamma, lmbda, mode='approx', return_sum_weight_value=False):

    reward = reward.to(torch.float64)
    if value is not None:
        value = value.to(torch.float64)
    next_value = next_value.to(torch.float64)
    # follow https://arxiv.org/pdf/1506.02438.pdf
    if mode != 'exact':
        assert not return_sum_weight_value
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

                lmbda_sum = 0.
                not_truncated = 1.0
                lastA = 0.
                for j in range(i, len(reward)):

                    R = R + reward[j] * discount_gamma

                    mask_done = (1. - done[j].float())[..., None]
                    A = R + (discount_gamma * gamma) * next_value[j] * mask_done - value[i] # done only stop future rewards ..

                    lmbda_sum += discount_lmbda

                    lastA = A * not_truncated + (1-not_truncated) * lastA
                    adv += (A * discount_lmbda) 

                    mask_truncated = (1. - truncated[j].float())[..., None] # mask truncated will stop future computation.

                    discount_gamma = discount_gamma * mask_truncated
                    discount_lmbda = discount_lmbda * mask_truncated
                    not_truncated = not_truncated * mask_truncated

                    # note that we will count done; always ...

                    discount_gamma = discount_gamma * gamma
                    discount_lmbda = discount_lmbda * lmbda

                #adv = adv/ lmbda_sum # normalize it based on the final result.
                adv = (adv + lastA  * (1./ (1.-lmbda) - lmbda_sum)) * (1-lmbda)
            else:
                raise NotImplementedError

            gae.append(adv)
    else:
        """
        1               -V(s_t)  + r_t                                                                                     + gamma * V(s_{t+1})
        lmabda          -V(s_t)  + r_t + gamma * r_{t+1}                                                                   + gamma^2 * V(s_{t+2})
        lambda^2        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                                               + ...
        lambda^3        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}

        We then normalize it by the sum of the lambda^i
        """
        sum_lambda = 0.
        sum_reward = 0.
        sum_end_v = 0.
        last_value = 0.
        gae = []
        mask_done = (1. - done.float())[..., None]
        mask_truncated = (1 - truncated.float())[..., None]
        if return_sum_weight_value:
            sum_weights = []
            last_values = []
            total = []

        for i in reversed(range(len(reward))):
            sum_lambda = sum_lambda * mask_truncated[i]
            sum_reward = sum_reward * mask_truncated[i]
            sum_end_v = sum_end_v * mask_truncated[i]

            sum_lambda = 1. + lmbda * sum_lambda
            sum_reward = lmbda * gamma * sum_reward + sum_lambda * reward[i]

            next_v = next_value[i] * mask_done[i]
            sum_end_v =  lmbda * gamma * sum_end_v  + gamma * next_v

            last_value = last_value * mask_truncated[i] + next_v * (1-mask_truncated[i]) # if truncated.. use the next_value; other wise..
            last_value = last_value * gamma + reward[i]
            # if i == len(reward) - 1:
            #     print('during the last', sum_reward, gamma, next_value[i], mask_done[i], value[i])
            sumA = sum_reward + sum_end_v

            if return_sum_weight_value:
                sum_weights.append(sum_lambda)
                last_values.append(last_value)
                total.append(sumA)

            expected_value = (sumA + last_value  * (1./ (1.-lmbda) - sum_lambda)) * (1-lmbda)
            # gg = sumA / sum_lambda 
            gae.append(expected_value - (value[i] if value is not None else 0))

        if return_sum_weight_value:
            sum_weights = torch.stack(sum_weights[::-1])
            last_values = torch.stack(last_values[::-1])
            total = torch.stack(total[::-1])
            return total, sum_weights, last_values

        gae = gae[::-1]

    return torch.stack(gae).float() #* (1-lmbda) 


def compute_expected_value_by_hand(reward, done, truncated, next_z, next_a, gamma, lmbda):
    import tqdm
    values = []
    for i in tqdm.trange(len(reward)):

        R = 0
        discount_gamma = 1.
        discount_lmbda = 1.
        sum_lmbda = 0.

        expect = 0.
        for j in range(i, len(reward)):
            done_mask = (1. - done[j].float())[..., None]

            R = R + reward[j] * discount_gamma # total reward to j

            V1 = R + (discount_gamma * gamma) * next_z[j] * done_mask
            V2 = R + (discount_gamma * gamma) * next_a[j] * done_mask


            expect += V1 * discount_lmbda
            sum_lmbda += discount_lmbda
            discount_lmbda = discount_lmbda * (lmbda ** 0.5)

            expect += V2 * discount_lmbda
            sum_lmbda += discount_lmbda
            discount_lmbda = discount_lmbda * (lmbda ** 0.5)

            truncated_mask = (1. - truncated[j].float())[..., None]
            discount_gamma = discount_gamma * truncated_mask
            discount_lmbda = discount_lmbda * truncated_mask

            discount_gamma = discount_gamma * gamma

        values.append(expect/sum_lmbda)

    return torch.stack(values)