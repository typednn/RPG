import tqdm
import numpy as np
import torch


def sample_batch(index, batch_size):
    return np.array_split(np.array(index), max(len(index)//batch_size, 1))


def minibatch_gen(traj, index, batch_size, KEY_LIST=None, verbose=False):
    if KEY_LIST is None:
        KEY_LIST = list(traj.keys())

    tmp = sample_batch(np.random.permutation(len(index)), batch_size)
    if verbose:
        tmp = tqdm.tqdm(tmp, total=len(tmp))
    for idx in tmp:
        k = index[idx]
        yield {
            key: [traj[key][i][j] for i, j in k]
            for key in KEY_LIST
        }


# traj is [nsteps x nenv x dict(key: val)]
def get_traj_index(traj) -> torch.Tensor:
    timesteps, nenv = traj['timesteps'], traj['nenv']
    return np.array([(i, j) for j in range(nenv) for i in range(timesteps)])


def predict_traj_value(
    traj, key, network, batch_size,
    index=None, vpred=None
) -> torch.Tensor:
    if isinstance(key, str):
        key = [key]
    assert isinstance(traj, list) and isinstance(traj[0], dict), "traj must be a list of dicts"

    timesteps = traj['timesteps']
    nenv = traj['nenv']
    index = index or get_traj_index(traj)

    if vpred is None:
        vpred = np.zeros((timesteps, nenv), dtype=np.float32)

    for ind in sample_batch(index, batch_size):
        obs = [[traj[i][k][j] for i, j in ind] for k in key]
        value = network(*obs)

        if vpred is None:
            vpred = torch.zeros((timesteps, nenv, *value.shape[1:]), device=value.device, dtype=value.dtype)

        ind = torch.tensor(ind, dtype=torch.long, device=value.device)
        vpred[ind[0], ind[1]] = value

    return vpred

def convert_traj_key_to_tensor(traj, key):
    from tools.utils import dconcat, dstack
    return dstack([dstack(traj[key][i]) for i in traj])