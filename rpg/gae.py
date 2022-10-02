# compute the value for 
import torch
import numpy as np


def compute_gae(
    vpred,
    next_vpred,
    reward,
    done,
    gamma,
    lmbda,
    _norm_adv,
    _epsilon,
    rew_rms,
):
    if isinstance(vpred, list):
        vpred = np.array(vpred)
        reward = np.array(reward)

    if rew_rms is not None:
        vpred = vpred * rew_rms.std
        next_vpred = next_vpred * rew_rms.std

    nstep = len(vpred)
    if isinstance(vpred[0], np.ndarray):
        adv = np.zeros((nstep, len(vpred[0])), np.float32)
    else:
        adv = torch.zeros((nstep, len(vpred[0])), device=vpred.device, dtype=vpred.dtype)
    lastgaelam = 0

    for t in reversed(range(nstep)):
        nextvalue = next_vpred[t]
        mask = 1. - done[t]
        assert mask.shape == nextvalue.shape
        #print(reward.device, next_vpred.device, mask.device, vpred[t].device)
        delta = reward[t] + gamma * nextvalue * mask - vpred[t]
        adv[t] = lastgaelam = delta + gamma * lmbda * lastgaelam * mask
        # nextvalue = vpred[t]

    vtarg = vpred + adv
    #reward = reward

    if rew_rms is not None:
        rew_rms.update(vtarg.reshape(-1))
        # https://github.com/haosulab/pyrl/blob/926d3d07d45f3bf014e7c6ea64e1bba1d4f35f03/pyrl/utils/torch/module_utils.py#L192
        vtarg = vtarg / rew_rms.std

    logger.logkvs({
        'reward_mean': reward.mean(),
        'reward_std': reward.std(),
        'vtarg_mean': vtarg.mean(),
        'orig_vpred_mean': vpred.mean(),
        'orig_vpred_std': vpred.std(),  # NOTE: this is denormalized ..
    })

    if _norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + _epsilon)
        logger.logkvs({
            'adv_normed_mean': adv.mean(),
            'adv_normed_std': adv.std(),
        })

    return vtarg, adv