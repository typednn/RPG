import torch
from torch import nn
from .dist_head import DistHead, ActionDistr
import numpy as np
from torch.distributions import Normal as Gaussian
# from .gmm import GMMAction


class NormalAction(ActionDistr):
    def __init__(self, loc, scale, tanh=False):
        self.dist = Gaussian(loc, scale)
        self.tanh = tanh
        self.kwargs = {'tanh': tanh}

    def rsample(self, detach=False):
        action = self.dist.rsample()
        if detach:
            action = action.detach()

        logp = self.dist.log_prob(action)
        if self.tanh:
            action = torch.tanh(action)
            logp -= torch.log(1. * (1 - action.pow(2)) + 1e-6)
        logp = logp.sum(axis=-1)
        return action, logp

    def entropy(self):
        return self.dist.entropy().sum(axis=-1)

    def sample(self):
        return self.rsample(detach=True)

    def get_parameters(self):
        return self.dist.loc, self.dist.scale

    def log_prob(self, action):
        if self.tanh:
            raise NotImplementedError
        return self.dist.log_prob(action).sum(axis=-1)

    def render(self, print_fn):
        print_fn('loc:', self.dist.loc.detach().cpu().numpy(), 'std:', self.dist.scale.detach().cpu().numpy(), 'entropy:', self.dist.entropy().detach().cpu().numpy())


class Normal(DistHead):
    LOG_STD_MAX = 10
    LOG_STD_MIN = -20

    STD_MODES = ['statewise', 'fix_learnable', 'fix_no_grad']

    def __init__(self,
                 action_space,
                 cfg=None,
                 std_mode: str = 'fix_no_grad',
                 std_scale=0.1,
                 minimal_std_val=-np.inf,
                 use_gmm=0,
                 squash=False, linear=True, nocenter=False):
        super().__init__(action_space)

        self.std_mode = std_mode
        self.std_scale = std_scale  # initial std scale
        self.minimal_std_val = minimal_std_val

        assert std_mode in self.STD_MODES
        n_output = 2 if std_mode == 'statewise' else 1

        self.net_output_dim = self.action_dim * n_output
        if use_gmm:
            self.net_output_dim = (n_output * self.action_dim + 1) * use_gmm

        if std_mode.startswith('fix'):
            self.log_std = nn.Parameter(torch.zeros(
                1, self.action_dim), requires_grad=('learnable' in std_mode))
        else:
            self.log_std = None

    def get_input_dim(self):
        return self.net_output_dim

    def forward(self, means):
        if self._cfg.use_gmm > 0:
            raise NotImplementedError
            log_loc = means[:, :self._cfg.use_gmm]
            means = means[:, self._cfg.use_gmm:]
            means = means.reshape(means.shape[0], self._cfg.use_gmm, -1)

        if self.std_mode.startswith('fix'):  # fix, determine
            log_stds = self.log_std.expand_as(means)
        else:  # 'tanh',
            means, log_stds = torch.chunk(means, 2, dim=-1)

        from tools.utils import clamp
        log_stds = clamp(
            log_stds, minval=max(self.LOG_STD_MIN, self.minimal_std_val), maxval=self.LOG_STD_MAX)

        action_std = torch.exp(log_stds) * self.std_scale
        assert not torch.isnan(means).any()

        if not self._cfg.linear:
            if not self._cfg.squash:
                means = torch.tanh(means)
                tanh = False
            else:
                tanh = True
        else:
            tanh = False  # linear gaussian ..

        if self._cfg.nocenter:
            means = means * 0

        if self._cfg.use_gmm == 0:
            return NormalAction(means, action_std, tanh=tanh)
        else:
            raise NotImplementedError
            #return GMMAction(log_loc, means, action_std, tanh=tanh)
