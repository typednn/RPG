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



        
class RNDExplorer(ExplorationBonus):
    def __init__(
        self,
        obs_space,
        state_dim,
        buffer,
        enc_s,
        z_space,
        cfg=None,
        normalizer='ema',
        as_reward=True, training_on_rollout=False,
        include_latent=False,
    ):


        self.inp_dim = inp_dim
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

        

from tools.utils import mlp, logger
class VAEExplorer(ExplorationBonus):
    def __init__(self, module, buffer, enc_s, cfg=None, buffer_size=None, update_step=1, update_freq=1, batch_size=512, obs_mode='state', normalizer=None, as_reward=True, training_on_rollout=False, scale=0, include_latent=False) -> None:
        super().__init__(module, buffer, enc_s, cfg, buffer_size, update_step, update_freq, batch_size, obs_mode, normalizer, as_reward, training_on_rollout, scale, include_latent)
    def make_autoencoder(self):
        self.encoder = mlp(self.obs_space.shape[0], [256, 512, 512, 256], self.latent.get_input_dim())
        self.decoder = mlp(self.latent.embed_dim(), [256, 512, 512, 256], self.obs_space.shape[0])