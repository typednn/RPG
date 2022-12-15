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
        
    def forward(self, x, hidden, timestep):
        return self.main(x)


# Positional encoding (section 5.1)
class Embedder:
    # https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class RNDOptim(OptimModule):
    def __init__(self, obs_space, state_dim, enc_s, cfg=None, use_embed=0, learning_epoch=1, mode='step', normalizer=True, rnd_scale=0.,):
        from .rnd import RNDNet
        # inp_dim = obs_space.shape[0] # TODO support other observation ..
        inp_dim = state_dim
        self.inp_dim = inp_dim
        if use_embed > 0:
            self.embeder, self.inp_dim = get_embedder(use_embed)
        else:
            self.embeder = None

        network = RNDNet(self.inp_dim).cuda()

        super().__init__(network, cfg)
        self.target = RNDNet(network.inp_dim, cfg=network._cfg).cuda()
        for param in self.target.parameters():
            param.requires_grad = False
        self.enc_s = enc_s

        self.normalizer = RunningMeanStd(last_dim=True) if normalizer else None
        self.as_reward = False

        assert rnd_scale == 1. or rnd_scale == 0.
        

    def compute_loss(self, obs, hidden, timestep, reduce=True):
        inps = obs

        if self.embeder is not None:
            inps = self.embeder(inps)

        from tools.utils import totensor
        inps = totensor(inps, device='cuda:0')

        predict = self.network(inps, hidden, timestep)
        with torch.no_grad():
            target =self.target(inps, hidden, timestep)

        loss = ((predict - target)**2).sum(axis=-1, keepdim=True)
        #assert loss.dim() == 2
        if reduce:
            loss = loss.mean()

        return loss


    def intrinsic_reward(self, rollout):
        loss = self.compute_loss(rollout['state'][1:], hidden=None, timestep=None, reduce=False)

        if self.normalizer is not None:
            loss = self.normalizer.normalize(loss)
            

        return 'rnd', loss

    def update_intrinsic(self, rollout):
        from .utils import flatten_obs
        all_obs = rollout['state'].detach()
        loss = self.compute_loss(all_obs, hidden=None, timestep=None, reduce=False)

        with torch.no_grad():
            if self.normalizer is not None:
                self.normalizer.update(loss)

        self.optimize(loss.mean())


    def visualize_transition(self, transition):
        attrs = transition.get('_attrs', {})
        if 'r' not in attrs:
            from tools.utils import tonumpy
            attrs['r'] = tonumpy(transition['r'])[..., 0] # just record one value ..

        attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_state']))
        transition['_attrs'] = attrs

    def compute_bonus(self, state):
        return self.compute_loss(state, hidden=None, timestep=None, reduce=False)

        
        
        
class RNDExplorer(ExplorationBonus):
    def __init__(
        self, obs_space, state_dim, buffer, enc_s,
        cfg=None, use_embed=0,
        normalizer='ema',
        as_reward=False, training_on_rollout=True
    ):

        inp_dim = state_dim
        self.inp_dim = inp_dim
        if use_embed > 0:
            self.embeder, self.inp_dim = get_embedder(use_embed)
        else:
            self.embeder = None
        network = RNDNet(self.inp_dim).cuda()

                 
        ExplorationBonus.__init__(self, network, buffer, enc_s)
        self.target = RNDNet(network.inp_dim, cfg=network._cfg).cuda()

        for param in self.target.parameters():
            param.requires_grad = False
        self.enc_s = enc_s

    def compute_loss(self, obs, hidden, timestep):
        inps = obs

        if self.embeder is not None:
            inps = self.embeder(inps)

        from tools.utils import totensor
        inps = totensor(inps, device='cuda:0')

        predict = self.network(inps, hidden, timestep)
        with torch.no_grad():
            target =self.target(inps, hidden, timestep)

        loss = ((predict - target)**2).sum(axis=-1, keepdim=True)
        return loss

    def compute_bonus(self, obs) -> torch.Tensor:
        loss =  self.compute_loss(obs, hidden=None, timestep=None)
        return loss



    def update(self, obs):
        from .utils import flatten_obs
        #loss = self.compute_loss(obs, hidden=None, timestep=None, reduce=False)
        loss = self.compute_bonus(obs)
        self.optimize(loss.mean())
        return loss.detach()