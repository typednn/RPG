# https://github.com/hzaskywalker/Concept/blob/0f5e95be1789e603d0a6f6bfdcec07d388be7bba/solver/diff_agent.py

import torch
import numpy as np
from torch import nn
from tools.nn_base import Network
from tools.optim import OptimModule
from tools.utils import RunningMeanStd, batch_input, logger
from .traj import DataBuffer, Trajectory
#from ..networks import concat
from .exploration_bonus import ExplorationBonus
from tools.nn_base import concat


class RNDNet(Network):
    def __init__(self, inp_dim, cfg=None, n_layers=3, dim=512):
        super(RNDNet, self).__init__()
        self.inp_dim = inp_dim
        layers = []
        for i in range(n_layers):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(inp_dim, dim))
            inp_dim = dim
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)

        
class RNDExplorer(ExplorationBonus):
    def __init__(
        self, obs_space, state_dim, buffer, enc_s, z_space,
        cfg=None,
        normalizer='ema',
        as_reward=False, training_on_rollout=True,

        include_latent=False,
    ):

        if cfg.obs_mode == 'state':
            inp_dim = state_dim
        elif cfg.obs_mode == 'obs':
            inp_dim = obs_space.shape[0]
        else:
            raise NotImplementedError
        if include_latent:
            inp_dim += z_space.dim
        self.inp_dim = inp_dim
        # if include_latent:
        #     print(z_space.dim)
        #     raise NotImplementedError
        self.z_space = z_space


        network = RNDNet(self.inp_dim).cuda()
        ExplorationBonus.__init__(self, network, buffer, enc_s)

        self.target = RNDNet(network.inp_dim, cfg=network._cfg).cuda()
        for param in self.target.parameters():
            param.requires_grad = False
        self.enc_s = enc_s

    def compute_loss(self, obs, latent):
        inps = obs
        from tools.utils import totensor
        inps = totensor(inps, device='cuda:0')
        if self.include_latent:
            latent = self.z_space.tokenize(totensor(latent, device='cuda:0'))
            inps = torch.cat([inps, latent], dim=-1)

        predict = self.network(inps)
        with torch.no_grad():
            target = self.target(inps)

        loss = ((predict - target)**2).sum(axis=-1, keepdim=True)
        return loss

    def compute_bonus(self, obs, latent) -> torch.Tensor:
        loss =  self.compute_loss(obs, latent)
        return loss


    def update(self, obs, latent):
        loss = self.compute_bonus(obs, latent)
        self.optimize(loss.mean())
        logger.logkv_mean(self.name + '_loss', loss.mean())
        return loss.detach()