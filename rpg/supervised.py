import matplotlib.pyplot as plt
import torch
import numpy as np  
from tools.utils import totensor
from tools.config import Configurable
from rpg.density import DensityEstimator

"""
# supervised learning for density measurement
# see notion here: https://www.notion.so/Density-Measure-fa57fc6eda8c4b94b9a8e37bb1490bd8

Task list
    1. dataset
        - simple Gaussian
        - dataset from rpg (buffer of test_new_triple)
        - ant buffer, and we change the ratio of each grid
    2. evaluation metrics
    3. model and optimizer
        - adhoc: return an occupancy map (with anchor )
 
    4. visualization
"""

class DatasetBase(Configurable):
    def __init__(self, cfg=None) -> None:
        super().__init__()

    def get_obs_space(self):
        raise NotImplementedError

    def sample(self, batch_size):
        # sample dataset
        raise NotImplementedError

    def count(self, inp):
        # return occupancy
        raise NotImplementedError

    def visualize(self, occupancy):
        # visualize the occupancy
        raise NotImplementedError

    def test(self):
        plt.clf()
        data = self.sample(1000)
        occupancy = self.count(data) 
        self.visualize(occupancy)
        plt.savefig('test.png')
        

class GaussianDataset(DatasetBase):
    def __init__(self, cfg=None, N=100) -> None:
        super().__init__()
        self.N = N
        self.bins = np.linspace(-5, 5., self.N)

    def sample(self, batch_size):
        n1 = int(batch_size * 0.3)
        X = np.concatenate(
            (np.random.normal(-1, 1, n1), np.random.normal(3, 0.3, batch_size - n1))
        )[:, np.newaxis]
        return totensor(X, device='cuda:0').clip(-5, 4.999999)

    def count(self, inp):
        inp = inp.reshape(-1)
        # print(plt.hist(inp.cpu().numpy(), bins=100)[1].shape)
        inp = ((inp / 10 + 0.5) * self.N).long()
        count = torch.bincount(inp, minlength=self.N)
        return count

    def visualize(self, occupancy):
        #return super().visualize(occupancy)
        occupancy = occupancy.detach().cpu().numpy()
        bins = np.append(self.bins, self.bins[-1] + 10. / self.N)
        plt.stairs(occupancy, bins, fill=True, label='occupancy')


class Env2DDataset(DatasetBase):
    def __init__(self, cfg=None, path='tmp/new/buffer.pt', N=25) -> None:
        super().__init__()
        with torch.no_grad():
            buffer = torch.load(path)
            self.data = buffer._next_obs[:buffer.total_size()].cpu().numpy()
        self.N = N
        self.xedges = np.linspace(0., 1, self.N + 1)
        self.yedges = np.linspace(0., 1, self.N + 1)

    def sample(self, batch_size):
        idx = np.random.choice(self.data.shape[0], batch_size)
        return totensor(self.data[idx], device='cuda:0')

    def count(self, inp):
        inp = inp[..., :2].reshape(-1, 2).detach().cpu().numpy()
        # plt.scatter(inp[:, 0], inp[:, 1])
        # plt.savefig('test.png')
        # exit(0)
        count, xedges, yedges = np.histogram2d(
            inp[:, 1], inp[:, 0], bins=self.N, range=[[0., 1], [0., 1]]
        )
        #plt.hist2d(inp[:, 0], inp[:, 1], bins=self.N, range=[[-1., 1], [-1., 1]])
        return totensor(count, device='cuda:0')

    def visualize(self, occupancy):
        count2d = occupancy.detach().cpu().numpy()
        plt.pcolormesh(self.xedges, self.yedges, count2d, shading='auto')
        pass


def make_dataset(dataset_name):
    if dataset_name == 'twonormal':
        return GaussianDataset()
    elif dataset_name == 'env2d':
        return Env2DDataset()
    else:
        raise NotADirectoryError



class Trainer(Configurable): 
    def __init__(self, cfg=None, dataset_name=None, 
                 density=DensityEstimator.to_build("RND")) -> None:
        super().__init__()
        self.dataset = make_dataset(dataset_name)
        self.density = DensityEstimator.to_build(self.dataset.get_obs_space(), cfg=density)

    def test_dataset(self):
        self.dataset.test()

if __name__ == '__main__':
    trainer = Trainer.parse(dataset_name='twonormal')
    trainer.test_dataset()