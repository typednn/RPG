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

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        """
        Potential (Anyway, this is just different way of value estimate.)
            V(s, prevz)
            Q(s, z): and then use the average of Q(s, z) to estimate V(s, prevz)
            Q(s, z, a): use pure sampling based method to estiamte
            Note that the entropy of the current step is not included ..
        """
        # return the Q value .. if it's value, return self._cfg.gamma
        # assert torch.allclose(a, z)
        z = self.enc_z(z)
        if self.action_dim > 0:
            inp = self.add_alpha(s, a, z)
        else:
            assert torch.allclose(a, z)
            inp = self.add_alpha(s, z)

        q1 = self.q(inp)
        q2 = self.q2(inp)
        value = torch.cat((q1, q2), dim=-1)
        return value, None

class ValuePolicy(AlphaPolicyBase):
    def __init__(
        self,state_dim, action_dim, z_space, enc_z, hidden_dim, cfg = None,
        zero_done_value=False,
    ) -> None:
        #nn.Module.__init__(self)
        AlphaPolicyBase.__init__(self)
        self.z_space = z_space
        self.enc_z = enc_z
        assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim + enc_z.output_dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim + enc_z.output_dim, hidden_dim, 1)

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        """
        Potential (Anyway, this is just different way of value estimate.)
            V(s, prevz)
            Q(s, z): and then use the average of Q(s, z) to estimate V(s, prevz)
            Q(s, z, a): use pure sampling based method to estiamte
            Note that the entropy of the current step is not included ..
        """
        # return the Q value .. if it's value, return self._cfg.gamma
        z = self.enc_z(z)
        mask = 1. if done is None else (1-done.float())
        inp = self.add_alpha(new_s, z)


        v1, v2 = self.q(inp), self.q2(inp)
        values = torch.cat((v1, v2), dim=-1)
        if self._cfg.zero_done_value:
            mask = 1. # ignore the done in the end ..
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
    def __init__(self, state_dim, hidden_dim, enc_z, cfg=None, K=1, output_ent=False, epsilon=0.) -> None:
        super().__init__()
        self.K = K
        self.enc_z = enc_z
        self.zdim = self.enc_z.output_dim
        self.qnet = self.build_backbone(
            state_dim, hidden_dim, (3 if output_ent else  1, self.zdim)
        )

    def q_value(self, state):
        v = self.qnet(self.add_alpha(state))
        return v[..., 0, :]

    def forward(self, state, prevz, timestep, z=None):
        new_action = timestep % self.K == 0
        new_action_prob = torch.zeros_like(new_action).float()

        q = self.q_value(state)
        logits = q / self.alpha[1]

        # v = self.alpha[1] * torch.log(torch.sum(torch.exp(q/self.alpha[1]), dim=1, keepdim=True))
        # dist = torch.exp((q-v)/self.alpha[1])
        # #dist = dist / torch.sum(dist)
        # print(torch.sum(dist))


        if self._cfg.epsilon  == 0.:
            pi_z = torch.distributions.Categorical(logits=logits) # the second is the z..
        else:
            prob = torch.softmax(logits, dim=-1) * (1 - self._cfg.epsilon) + self._cfg.epsilon / logits.shape[-1]
            pi_z = torch.distributions.Categorical(probs=prob)
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
        entropies = rollout['extra_rewards'].detach()

        # replicate the q value to the pi_z ..
        q_value = q_value.min(axis=-1, keepdims=True).values
        with torch.no_grad():
            q_target = q_value + entropies[..., 0:1] + entropies[..., 2:3]

        q_val = self.q_value(state[:-1])
        q_predict = batch_select(q_val, z)
        assert q_predict.shape == q_target.shape
        # assert torch.allclose(q_predict, batch_select(self.q_value(state), z))
        return ((q_predict - q_target)**2).mean() # the $z$ selected should be consistent with the policy ..  