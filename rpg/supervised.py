import matplotlib.pyplot as plt
import torch
import numpy as np  
from tools.utils import totensor
from tools.config import Configurable

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
        data = self.sample(1000)
        occupancy = self.count(data) 
        plt.clf()
        self.visualize(occupancy)
        plt.savefig('test.png')
        

class GaussianDataset(DatasetBase):
    def __init__(self, cfg=None, N=100) -> None:
        super().__init__()
        self.N = N
        self.bins = np.linspace(-5, 5, self.N)

    def sample(self, batch_size):
        n1 = int(batch_size * 0.3)
        X = np.concatenate(
            (np.random.normal(0, 1, n1), np.random.normal(5, 1, batch_size - n1))
        )[:, np.newaxis]
        return totensor(X, device='cuda:0').clip(-5, 5)

    def count(self, inp):
        return ((inp / 10 + 0.5) * self.N).long()

    def visualize(self, occupancy):
        return super().visualize(occupancy)

def make_dataset(dataset_name):
    if dataset_name == 'twonormal':
        return GaussianDataset()
    else:
        raise NotADirectoryError



class Trainer(Configurable): 
    def __init__(self, cfg=None, dataset_name=None) -> None:
        super().__init__()
        self.dataset = make_dataset(dataset_name)

    def test_dataset(self):
        self.dataset.test()

if __name__ == '__main__':
    trainer = Trainer(dataset_name='twonormal')
    trainer.test_dataset()