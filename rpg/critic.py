import torch
import numpy as np
from torch import nn
import torch.optim
from tools.config import Configurable, as_builder
import torch as th
from tools.utils import logger, batch_input
from nn.distributions import ActionDistr
from tools.optim import OptimModule as Optim


class Critic(Optim):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5, mode='step'):
        super(Critic, self).__init__(critic)
        self.critic = critic
        #self.optim = make_optim(critic.parameters(), lr)
        self.vfcoef = vfcoef

    def compute_loss(self, obs, old_z, z, vtarg_z, vtarg_a):
        vpred = self.critic(obs)[..., 0]
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf
