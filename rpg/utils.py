import tqdm
import numpy as np
import torch
from nn.space import Discrete, Box, MixtureSpace
from tools.config import Configurable



def iter_batch(index, batch_size):
    return np.array_split(np.array(index), max(len(index)//batch_size, 1))


def minibatch_gen(traj, index, batch_size, KEY_LIST=None, verbose=False):
    # traj is dict of [nsteps, nenv, datas.]
    if KEY_LIST is None:
        KEY_LIST = list(traj.keys())
    tmp = iter_batch(np.random.permutation(len(index)), batch_size)
    if verbose:
        tmp = tqdm.tqdm(tmp, total=len(tmp))
    for idx in tmp:
        k = index[idx]
        yield {
            key: [traj[key][i][j] for i, j in k] if traj[key] is not None else None
            for key in KEY_LIST
        }



def create_hidden_space(z_dim, z_cont_dim):
    if z_cont_dim == 0:
        z_space = Discrete(z_dim)
    elif z_dim == 0:
        z_space = Box(-1, 1, (z_cont_dim,))
    else:
        z_space = MixtureSpace(Discrete(z_dim), Box(-1, 1, (z_cont_dim,)))
    return z_space


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