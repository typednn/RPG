import torch
from torch import nn
from tools.nn_base import Network
from tools import dist_utils

class Optim(Network):
    KEYS = []
    name = None
    def __init__(
        self, network, cfg=None,
        lr=3e-4, max_grad_norm=None, eps=1e-8,
        accumulate_grad=0, mode='not_step',
    ):
        super().__init__()
        self.network = network
        dist_utils.sync_networks(self.network)
        self.params = list(network.parameters() if isinstance(network, nn.Module) else [network])
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr, eps=cfg.eps)

        if accumulate_grad > 1:
            self.n_steps = 0

    def optim_step(self):
        dist_utils.sync_grads(self.network)
        if self._cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.params, self._cfg.max_grad_norm)
        self.optimizer.step()

    def optimize(self, loss: torch.Tensor):
        if self._cfg.accumulate_grad > 1:
            if self.n_steps % self._cfg.accumulate_grad == 0:
                self.zero_grad()

            (loss / self._cfg.accumulate_grad).backward()
            if (self.n_steps+1) % self._cfg.accumulate_grad == 0:
                self.optim_step()

            self.n_steps += 1
        else:
            self.zero_grad()
            loss.backward()
            self.optim_step()