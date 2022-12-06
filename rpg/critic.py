# critic network
# https://www.notion.so/Model-based-RPG-Q-version-3c97a98eea3445968ef634f684f2d152

# from simplify the code, we only compute $Q$ but not separte the value output..
import torch
from torch import nn
from tools.utils import Seq, mlp, logger
from nn.distributions import CategoricalAction, Normal
from tools.nn_base import Network
from gym import spaces


class AlphaPolicyBase(Network):
    def __init__(self, cfg=None, observe_alpha=False) -> None:
        super().__init__()
        self.observe_alpha = observe_alpha
        self.alpha_dim = 2 if observe_alpha else 0
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def add_alpha(self, *args):
        x = torch.cat(args, dim=-1)
        if self.observe_alpha:
            v = torch.zeros(*x.shape[:-1], len(self.alphas), device=x.device, dtype=x.dtype) + self.alphas
            x = torch.cat([x, v], dim=-1)
        return x

    def build_backbone(self, inp_dim, hidden_dim, output_shape):
        return mlp(inp_dim + self.alpha_dim, hidden_dim, output_shape)


def batch_select(values, z=None):
    if z is None:
        return values
    else:
        out = torch.gather(values, -1, z.unsqueeze(-1))
        #print(values[2, 10, z[2, 10]], out[2, 10])
        return out

class SoftQPolicy(AlphaPolicyBase):
    def __init__(
        self,state_dim, action_dim, z_space, enc_z, hidden_dim, cfg = None,
    ) -> None:
        #nn.Module.__init__(self)
        AlphaPolicyBase.__init__(self)
        self.z_space = z_space
        self.enc_z = enc_z
        assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim + action_dim + enc_z.output_dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim + action_dim + enc_z.output_dim, hidden_dim, 1)
        self.action_dim = action_dim

    from tools.utils import print_input_args

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        z = self.enc_z(z)
        if self.action_dim > 0:
            inp = self.add_alpha(s, a, z)
        else:
            assert torch.allclose(a, z)
            inp = self.add_alpha(s, z)

        q1 = self.q(inp)
        q2 = self.q2(inp)
        q_value = torch.cat((q1, q2), dim=-1)
        return q_value, None

    def get_predict(self, rollout):
        return rollout['q_value']

    def compute_target(self, vtarg, reward, done_gt, gamma):
        return reward + (1-done_gt.float()) * gamma * vtarg


class ValuePolicy(AlphaPolicyBase):
    def __init__(
        self,state_dim, action_dim, z_space, enc_z, hidden_dim, cfg = None,
        zero_done_value=True,
    ) -> None:
        #nn.Module.__init__(self)
        AlphaPolicyBase.__init__(self)
        self.z_space = z_space
        self.enc_z = enc_z
        # assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim + enc_z.output_dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim + enc_z.output_dim, hidden_dim, 1)

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        # return the Q value .. if it's value, return self._cfg.gamma
        z = self.enc_z(z)
        mask = 1. if done is None else (1-done.float())
        # print(new_s.shape, new_s.device, z.shape, z.device)
        inp = self.add_alpha(new_s, z)

        v1, v2 = self.q(inp), self.q2(inp)
        values = torch.cat((v1, v2), dim=-1)
        if self._cfg.zero_done_value:
            mask = 1. # ignore the done in the end ..
        q_value = values * gamma * mask + r
        return q_value, values

    def get_predict(self, rollout):
        return rollout['pred_values']

    def compute_target(self, vtarg, reward, done_gt, gamma):
        if self._cfg.zero_done_value:
            vtarg = vtarg * (1 - done_gt.float())
        return vtarg