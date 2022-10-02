import torch
from nn.distributions import ActionDistr
from solver.actor import Actor as BaseNet
from gym.spaces import Box


class Policy(BaseNet):
    def __init__(self, obs_space, z_space, action_space, cfg=None):
        self.z_space = z_space
        if z_space is not None:
            obs_space = (obs_space, z_space)
        BaseNet.__init__(self, obs_space, action_space)

    def forward(self, state, hidden, timestep=None):
        if self.z_space is None:
            return super().forward(state, timestep=timestep)
        else:
            return super().forward((state, hidden), timestep=timestep)


class Critic(BaseNet):
    def __init__(self, obs_space, z_space, dim, cfg=None):
        self.z_space = z_space
        if z_space is not None:
            obs_space = (obs_space, z_space)
        BaseNet.__init__(obs_space, Box(-1, 1, (dim,)), head=dict(TYPE='Deterministic'))

    def forward(self, state, hidden, timestep=None):
        if self.z_space is None:
            return super().forward(state, timestep=timestep)
        else:
            return super().forward((state, hidden), timestep=timestep)