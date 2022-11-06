# critic network
# https://www.notion.so/Model-based-RPG-Q-version-3c97a98eea3445968ef634f684f2d152

# from simplify the code, we only compute $Q$ but not separte the value output..
import torch
from torch import nn
from tools.utils import Seq, mlp, logger
from nn.distributions import CategoricalAction
from tools.nn_base import Network
from gym import spaces



class AlphaPolicyBase(Network):
    def __init__(self, observe_alpha, cfg=None) -> None:
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
        from tools.utils import batched_index_select
        return batched_index_select(values, len(values.shape)-1, z)

class SoftQPolicy(AlphaPolicyBase):
    def __init__(
        self,
        state_dim, action_dim, z_dim, hidden_dim,
        cfg = None,
        observe_alpha=False,
    ) -> None:
        nn.Module.__init__(self)
        self.observe_alpha = observe_alpha
        assert z_dim != 0 
        self.q = self.build_backbone(state_dim + action_dim, hidden_dim, z_dim)
        self.q2 = self.build_backbone(state_dim + action_dim, hidden_dim, z_dim)

    def Q(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        """
        Potential
            V(s, prevz)
            Q(s, z): and then use the average of Q(s, z) to estimate V(s, prevz)
            Q(s, z, a): use pure sampling based method to estiamte

            Anyway, this is just different way of value estimate.

            Note that the entropy of the current step is not included ..
        """
        # return the Q value .. if it's value, return self._cfg.gamma
        inp = self.add_alpha(s, a)
        q_values = self.q(inp)
        q_values2 = self.q2(inp)
        return torch.cat((batch_select(q_values, z), batch_select(q_values2, z)), dim=-1)


from tools.config import Configurable, as_builder
from collections import namedtuple

Zout = namedtuple('Zout', ['z', 'logp_z', 'new', 'logp_new', 'entropy'])
Aout = namedtuple('Aout', ['a', 'logp_a'])

class PolicyA(AlphaPolicyBase):
    def __init__(self, state_dim, hidden_dim, enc_hidden, head, cfg=None, observe_alpha=False, mode='gd') -> None:
        super().__init__(observe_alpha)
        self.head = head
        self.enc_hidden = enc_hidden
        self.mode = mode
        self.backbone = self.build_backbone(
            state_dim + enc_hidden.output_dim, hidden_dim, head.get_input_dim())

    def forward(self, state_emebd, hidden):
        inp = self.add_alpha(state_emebd, self.enc_hidden(hidden))
        dist = self.head(self.backbone(inp))
        if self.mode == 'gd':
            return Aout(*dist.rample())
        else:
            raise NotImplementedError

    def loss(self, rollout):
        return -rollout[0]['total_value']


class SoftPolicyZ(AlphaPolicyBase):
    def __init__(self, state_dim, hidden_dim, z_dim, cfg=None, K=1, output_ent=False) -> None:
        super().__init__()
        self.K = K
        self.enc_hidden = z_dim
        self.zdim = self.enc_hidden.output_dim
        self.qnet = self.build_backbone(
            state_dim, hidden_dim, (3 if output_ent else  1, z_dim)
        )

    def q_value(self, state):
        return self.qnet(self.add_alpha(state))

    def forward(self, state, prevz, timestep, z=None):
        # soft Q policy for learning z ..
        new_action = timestep % self.K == 0
        new_action_prob = torch.zeros_like(new_action).float()
        #raise NotImplementedError

        pi_z = torch.distributions.Categorical(self.q_value(state) / self.alpha[1]) # the second is the z..
        z = pi_z.sample()
        logp_z = pi_z.log_prob(z)

        z = torch.where(new_action, prevz, z)
        logp_z = torch.where(new_action, new_action_prob, logp_z)
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
            q_target = q_value + entropies['a'][:, 0]
        q_predict = batch_select(self.q_value(state), z)
        return (q_predict - q_target).mean() # the $z$ selected should be consistent with the policy ..  