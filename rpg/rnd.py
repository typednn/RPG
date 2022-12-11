# https://github.com/hzaskywalker/Concept/blob/0f5e95be1789e603d0a6f6bfdcec07d388be7bba/solver/diff_agent.py

import torch
import numpy as np
from torch import nn
from tools.nn_base import Network
from tools.optim import OptimModule
from tools.utils import RunningMeanStd, batch_input, logger
from .traj import DataBuffer, Trajectory
#from ..networks import concat
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
    KEYS = ['obs']
    name = 'rnd'

    def __init__(self, obs_space, state_dim, cfg=None, use_embed=0, learning_epoch=1, mode='step', normalizer=True, rnd_scale=0.,):
        from .rnd import RNDNet
        # inp_dim = obs_space.shape[0] # TODO support other observation ..
        inp_dim = state_dim
        self.inp_dim = inp_dim
        if use_embed > 0:
            self.embeder, self.inp_dim = get_embedder(use_embed)
        else:
            self.embeder = None

        network = RNDNet(self.inp_dim).cuda()
        self.normalizer = RunningMeanStd(last_dim=False) if normalizer else None

        super().__init__(network, cfg)
        self.target = RNDNet(network.inp_dim, cfg=network._cfg).cuda()
        for param in self.target.parameters():
            param.requires_grad = False


    def __call__(self, traj: Trajectory, batch_size, update_normalizer=True):
        out = traj.predict_value(('obs', 'z', 'timestep'),
                                  lambda x, y, z: self.compute_loss(x, y, z, reduce=False), batch_size=batch_size) 

        if self.normalizer is not None:
            if update_normalizer:
                self.normalizer.update(out)
            out = self.normalizer.normalize(out)

        logger.logkvs_mean({'rnd/reward': out.mean().item()})
        return out * self._cfg.rnd_scale


    # def learn(self, data: DataBuffer, batch_size, logger_scope='rnd', **kwargs):
    #     for i in range(self._cfg.learning_epoch):
    #         n_batches = 0
    #         keys = ['obs', 'z', 'timestep']
    #         for batch in data.loop_over(batch_size, keys):
    #             n_batches += 1
    #             _, info = self.step(*[batch[i] for i in keys])
    #             if logger_scope is not None:
    #                 assert isinstance(logger_scope, str) and len(logger_scope) > 0
    #                 logger.logkvs_mean({f'{logger_scope}/'+k:v for k, v in info.items()})


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

    
    def plot2d(self):
        images = []

        for i in range(32):
            coords = (np.stack([np.arange(32), np.zeros(32)+i], axis=1) + 0.5)/32.
            out = self.forward_network(coords, update_norm=False)
            images.append(out.detach().cpu().numpy())

        return np.array(images)


    # def update_with_buffer(self, seg):
    #     #obs_seq = seg.obs_seq
    #     from .utils import flatten_obs
    #     all_obs = flatten_obs(seg.obs_seq)
    #     loss = self.compute_loss(all_obs, hidden=None, timestep=None)
    #     self.optimize(loss)
    #     # update the 

    def update_with_buffer(self, *args, **kwargs):
        pass

    def intrinsic_reward(self, rollout):
        loss = self.compute_loss(rollout['state'][1:], hidden=None, timestep=None, reduce=False)
        return 'rnd', loss

    def update_intrinsic(self, rollout):
        from .utils import flatten_obs
        all_obs = rollout['state'].detach()
        loss = self.compute_loss(all_obs, hidden=None, timestep=None, reduce=True)
        self.optimize(loss)