# critic network
# https://www.notion.so/Model-based-RPG-Q-version-3c97a98eea3445968ef634f684f2d152

# from simplify the code, we only compute $Q$ but not separte the value output..
import torch
from torch import nn
from tools.utils import Seq, mlp, logger
from nn.distributions import CategoricalAction
from tools.nn_base import Network
from gym import spaces
from tools.config import Configurable, as_builder
from collections import namedtuple


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
        self,state_dim, action_dim, z_space, hidden_dim, cfg = None,
    ) -> None:
        #nn.Module.__init__(self)
        AlphaPolicyBase.__init__(self)
        self.z_space = z_space
        assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim + action_dim, hidden_dim, z_space.n)
        self.q2 = self.build_backbone(state_dim + action_dim, hidden_dim, z_space.n)

    def forward(self, s, z, a_embed, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        """
        Potential (Anyway, this is just different way of value estimate.)
            V(s, prevz)
            Q(s, z): and then use the average of Q(s, z) to estimate V(s, prevz)
            Q(s, z, a): use pure sampling based method to estiamte
            Note that the entropy of the current step is not included ..
        """
        # return the Q value .. if it's value, return self._cfg.gamma
        inp = self.add_alpha(s, a_embed)
        q1 = self.q(inp)
        q2 = self.q2(inp)
        q_values = batch_select(q1, z)
        q_values2 = batch_select(q2, z)
        value = torch.cat((q_values, q_values2), dim=-1)
        return value, None

class ValuePolicy(AlphaPolicyBase):
    def __init__(
        self,state_dim, action_dim, z_space, enc_z, hidden_dim, cfg = None,
    ) -> None:
        #nn.Module.__init__(self)
        AlphaPolicyBase.__init__(self)
        self.z_space = z_space
        self.enc_z = enc_z
        assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim, hidden_dim, 1)

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        """
        Potential (Anyway, this is just different way of value estimate.)
            V(s, prevz)
            Q(s, z): and then use the average of Q(s, z) to estimate V(s, prevz)
            Q(s, z, a): use pure sampling based method to estiamte
            Note that the entropy of the current step is not included ..
        """
        # return the Q value .. if it's value, return self._cfg.gamma
        mask = 1. if done is None else (1-done.float())
        inp = self.add_alpha(new_s)

        v1, v2 = self.q(inp), self.q2(inp)
        values = torch.cat((v1, v2), dim=-1)
        return values * gamma * mask + r, values


Zout = namedtuple('Zout', ['z', 'logp_z', 'new', 'logp_new', 'entropy'])
Aout = namedtuple('Aout', ['a', 'logp_a'])

class PolicyA(AlphaPolicyBase):
    def __init__(self, state_dim, hidden_dim, enc_hidden, head, cfg=None, mode='gd'):
        super().__init__()
        self.head = head
        self.enc_hidden = enc_hidden
        self.mode = mode
        self.backbone = self.build_backbone(
            state_dim + enc_hidden.output_dim, hidden_dim, head.get_input_dim())

    def forward(self, state_emebd, hidden):
        inp = self.add_alpha(state_emebd, self.enc_hidden(hidden))
        dist = self.head(self.backbone(inp))

        from nn.distributions import NormalAction
        if isinstance(dist, NormalAction):
            scale = dist.dist.scale
            logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})

        if self.mode == 'gd':
            return Aout(*dist.rsample())
        else:
            raise NotImplementedError

    def loss(self, rollout):
        assert rollout['value'].shape[-1] == 2
        return -rollout['value'][..., 0].mean()


class SoftPolicyZ(AlphaPolicyBase):
    def __init__(self, state_dim, hidden_dim, enc_z, cfg=None, K=1, output_ent=False) -> None:
        super().__init__()
        self.K = K
        self.enc_z = enc_z
        self.zdim = self.enc_z.output_dim
        self.qnet = self.build_backbone(
            state_dim, hidden_dim, (3 if output_ent else  1, self.zdim)
        )

    def q_value(self, state):
        return self.qnet(self.add_alpha(state))[..., 0, :]

    def forward(self, state, prevz, timestep, z=None):
        # soft Q policy for learning z ..
        new_action = timestep % self.K == 0
        new_action_prob = torch.zeros_like(new_action).float()

        pi_z = torch.distributions.Categorical(logits=self.q_value(state) / self.alpha[1]) # the second is the z..
        z = pi_z.sample()
        logp_z = pi_z.log_prob(z)

        z = torch.where(new_action, z, prevz)
        logp_z = torch.where(new_action, logp_z, new_action_prob)
        entropy = pi_z.entropy() * new_action.float()

        return Zout(z, logp_z, new_action, new_action_prob, entropy)

    def loss(self, rollout):
        # for simplicity, we directly let the high-level policy to select the action with the best z values .. 
        state = rollout['state'].detach()
        q_value = rollout['q_value'].detach()
        z = rollout['z'].detach()
        entropies = rollout['entropies'].detach()

        # replicate the q value to the pi_z ..
        with torch.no_grad():
            q_target = q_value + entropies[..., 0:1]
        q_predict = batch_select(self.q_value(state), z)
        q_target = q_target.min(axis=-1, keepdims=True).values
        return ((q_predict - q_target)**2).mean() # the $z$ selected should be consistent with the policy ..  