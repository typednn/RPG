import torch
from tools import Configurable, as_builder
from tools.optim import OptimModule


@as_builder
class DensityEstimator(OptimModule):
    name='density'
    def __init__(self, space, cfg=None) -> None:
        # build the network
        network = self.make_network(space)
        super().__init__(network)
        self.network = network

    def make_network(self, space):
        raise NotImplementedError
        
    def log_prob(self, samples):
        # forward the network
        raise NotImplementedError

    def update(self, samples):
        # forward the network
        raise NotImplementedError