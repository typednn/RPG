import torch
import numpy as np
from tools import Configurable, as_builder
from tools.optim import OptimModule
from collections import deque
from tools.utils import RunningMeanStd


class ScalarNormalizer:
    # maintain a deque of fixed size and update the parameters with the recent data
    def __init__(self, size=10000):
        self.size = size
        self.data = deque()
        self.sum = 0.
        self.sumsq = 1.
        self.step = 0

    def update(self, data):
        sum = float(data.sum())
        sumsq = float((data**2).sum())
        self.data.append([sum, sumsq])
        self.sum += sum
        self.sumsq += sumsq
        if len(self.data)> self.size:
            sum, sumsq = self.data.popleft()
            self.sum -= sum
            self.sumsq -= sumsq

    @property
    def mean(self):
        return self.sum / len(self.data)

    @property
    def std(self):
        return np.sqrt(self.sumsq / len(self.data) - self.mean**2 + 1e-16)

    def normalize(self, data):
        return (data - self.mean) / self.std


@as_builder
class DensityEstimator(OptimModule):
    name='density'
    def __init__(self, space, cfg=None, normalizer=None) -> None:
        # build the network
        network = self.make_network(space)
        super().__init__(network)
        self.network = network


        if normalizer is None or normalizer == 'none':
            self.normalizer = None
        elif normalizer == 'ema':
            self.normalizer = RunningMeanStd(last_dim=True)
        elif isinstance(normalizer, int):
            self.normalizer = ScalarNormalizer(normalizer)
        else:
            raise NotImplementedError


    def make_network(self, space):
        raise NotImplementedError
        
    def _log_prob(self, samples):
        # forward the network
        raise NotImplementedError

    def _update(self, samples):
        # update; and return log prob for update
        raise NotImplementedError

    def update(self, samples):
        log_prob = self._update(samples)
        if self.normalizer is not None:
            self.normalizer.update(log_prob)

    def log_prob(self, samples):
        log_prob = self._log_prob(samples)
        if self.normalizer is not None:
            log_prob = self.normalizer.normalize(log_prob)
        return log_prob