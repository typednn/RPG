import abc
from tools.config import Configurable, as_builder


@as_builder
class Scheduler(Configurable, abc.ABC):
    def __init__(self, cfg=None, init_value=1.) -> None:
        super().__init__()

        self.epoch = 0
        self.init_value = init_value
        self.value = init_value

    def step(self, epoch=None):
        if epoch is None:
            self.epoch += 1
            delta = 1
        else:
            assert epoch > self.epoch
            delta = epoch - self.epoch
            self.epoch = epoch

        self.value = self._step(self.epoch, delta)
        return self.value
    
    def get(self):
        return self.value


    @abc.abstractmethod
    def _step(self, cur_epoch, delta):
        pass


class constant(Scheduler):
    def __init__(self, cfg=None) -> None:
        super().__init__()

    def _step(self, cur_epoch, delta):
        return self.init_value


class exp(Scheduler):
    def __init__(self, cfg=None, gamma=0.99, min_value=0.) -> None:
        super().__init__()
        self.gamma = gamma
        self.min_value = min_value

    def _step(self, cur_epoch, delta):
        return max(self.min_value, self.value * (self.gamma ** delta))


class stage(Scheduler):
    def __init__(self, cfg=None, milestones=None, gamma=0.1) -> None:
        super().__init__()
        self.milestones = milestones
        self.gamma = gamma

    def _step(self, cur_epoch, delta):
        value = self.init_value
        for i in self.milestones:
            if cur_epoch >= i:
                value = value * self.gamma
        return value