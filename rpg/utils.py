import tqdm
import numpy as np
import torch


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


