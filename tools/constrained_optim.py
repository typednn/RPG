import torch
import torch.nn as nn
from tools import dist_utils
from tools.config import Configurable
from .optim import OptimModule

class COptim(Configurable):
    def __init__(
        self,
        network,
        n_constraints,
        cfg=None,
        clip_lmbda=(0.1, 1e10),
        weight_penalty=0.001,
        max_grad_norm=None,
        reg_proj=0.01,
        constraint_threshold=0.,
        mu=1.,
        lr=0.001
    ) -> None:
        super().__init__()

        self.network = network

        dist_utils.sync_networks(network)
        self.params = list(
            network.parameters() if isinstance(network, nn.Module) else [network])

        self.actions_params = network
        self.loss_optim = torch.optim.Adam(self.params, lr=cfg.lr)
        self.constraint_optim = torch.optim.Adam(self.params, lr=cfg.lr)

        #TODO: change the optim ..
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(n_constraints, requires_grad=True, device=self.params[0].device))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.last_good = None

    def optimize_loss(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        dist_utils.sync_grads(params)
        if self._cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self._cfg.max_grad_norm)
        optim.step()

    def optimize(self, loss, C):
        assert C.shape[0] == self.log_alpha.shape[0]
        penalty = 0.
        mask = (C > 0.).float()

        constraints = (C * self.log_alpha.exp() + (C ** 2) * self._cfg.mu/2) * mask

        penalty = 0.
        reg_action = 0.
        c = constraints.sum()

        if c <= self._cfg.constraint_threshold:
            penalties = -torch.log(-c.clamp(min=1e-10)) * (1. - mask) 
            penalty = penalties.sum() * self._cfg.weight_penalty
            self.optimize_loss(self.loss_optim, self.params, loss + penalty)
        else:
            if self._cfg.reg_proj > 0. and self.last_good is not None:
                for a, b in zip(self.params, self.last_good):
                    reg_action += torch.sum((a - b) ** 2)
                reg_action = reg_action * self._cfg.reg_proj
            self.optimize_loss(self.constraint_optim, self.params, c + reg_action)

        alpha = self.log_alpha.exp()
        alpha_loss = torch.sum(self.log_alpha.exp() * C.detach()) # c later than 0, reduce log_alpha
        self.optimize_loss(self.alpha_optim, [self.log_alpha], alpha_loss)

        with torch.no_grad():
            if c <= 0.:
                self.last_good = [i.data.clone() for i in self.params]

        return {
            'penalty': penalty,
            'rec_action': float(reg_action),
            'c': float(c),
            'alpha': alpha.detach().cpu().numpy(),
        }