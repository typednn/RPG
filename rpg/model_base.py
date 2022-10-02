import torch
from nn.distributions import ActionDistr


class Policy:
    def __init__(self) -> None:
        pass

    def __call__(self, obs, hidden, timestep) -> ActionDistr:
        pass


class Critic:
    def __init__(self) -> None:
        pass

    def a(self, obs, hidden, timestep) -> torch.Tensor:
        pass

    def z(self, obs, hidden, timestep) -> torch.Tensor:
        pass
