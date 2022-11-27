# critic network
# https://www.notion.so/Model-based-RPG-Q-version-3c97a98eea3445968ef634f684f2d152

# from simplify the code, we only compute $Q$ but not separte the value output..
import torch
from torch import nn
from tools.utils import Seq, mlp, logger
from nn.distributions import CategoricalAction, Normal
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
        # assert isinstance(z_space, spaces.Discrete)
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
        # print(new_s.shape, new_s.device, z.shape, z.device)
        inp = self.add_alpha(new_s, z)

        v1, v2 = self.q(inp), self.q2(inp)
        values = torch.cat((v1, v2), dim=-1)
        if self._cfg.zero_done_value:
            mask = 1. # ignore the done in the end ..
        return values * gamma * mask + r, values


Aout = namedtuple('Aout', ['a', 'logp', 'ent'])

from nn.distributions import Normal, DistHead

@as_builder
class PolicyBase(AlphaPolicyBase):
    def __init__(self, cfg=None):
        super().__init__()

class DiffPolicy(PolicyBase):
    def __init__(self, state_dim, z_dim, hidden_dim, action_space,
                 cfg=None, head=Normal.gdc(), mode='gd'):
        super().__init__()

        head = DistHead.build(action_space, cfg=head)
        from .soft_actor_critic import DiffPolicy

        self.head = head
        self.mode = mode
        self.backbone = self.build_backbone(
            state_dim + z_dim,  hidden_dim,
            head.get_input_dim()
        )

    def forward(self, state_embed, hidden):
        inp = self.add_alpha(state_embed, self.enc_hidden(hidden))
        dist = self.head(self.backbone(inp))

        from nn.distributions import NormalAction
        if isinstance(dist, NormalAction):
            scale = dist.dist.scale
            logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})

        if self.mode == 'gd':
            a, logp = dist.rsample()
            return Aout(a, logp, -logp)
        else:
            raise NotImplementedError

    def loss(self, rollout):
        assert rollout['value'].shape[-1] == 2
        return -rollout['value'][..., 0].mean()


Zout = namedtuple('Zout', ['z', 'logp_z', 'new', 'logp_new', 'entropy'])
class HiddenPolicy(AlphaPolicyBase):
    def __init__(self, policy, cfg=None, K=1000000000) -> None:
        super().__init__()


    def forward(self, state, prevz, timestep):
        new_action = (timestep % self._cfg.K == 0)
        new_action_prob = torch.zeros_like(new_action).float()
        z = prevz.clone()


        logp_z = torch.zeros_like(new_action_prob)
        entropy = torch.zeros_like(new_action_prob) 

        if new_action.any():
            pi_z = self.policy(state[new_action])
            newz, newz_logp = pi_z.sample()
            z[new_action] = newz
            logp_z[new_action] = newz_logp
            entropy[new_action] = -newz_logp
        return Zout(z, logp_z, new_action, new_action_prob, entropy)



# class SoftPolicyZ(PolicyZ):
#     def __init__(self, state_dim, hidden_dim, enc_z, cfg=None, epsilon=0.) -> None:
#         super().__init__()
#         self.enc_z = enc_z
#         self.zdim = self.enc_z.output_dim
#         self.qnet = self.build_backbone(state_dim, hidden_dim, self.zdim)

#     def q_value(self, state):
#         return self.qnet(self.add_alpha(state))

#     def policy(self, state):
#         q = self.q_value(state)
#         logits = q / self.alpha[1]

#         from nn.distributions import CategoricalAction
#         out = CategoricalAction(logits, epsilon=self._cfg.epsilon)
#         return out

#     def loss(self, rollout):
#         # for simplicity, we directly let the high-level policy to select the action with the best z values .. 
#         state = rollout['state'].detach()
#         q_value = rollout['q_value'].detach()
#         z = rollout['z'].detach()
#         entropies = rollout['extra_rewards'].detach()

#         # replicate the q value to the pi_z ..
#         q_value = q_value.min(axis=-1, keepdims=True).values
#         with torch.no_grad():
#             q_target = q_value + entropies[..., 0:1] + entropies[..., 2:3]

#         q_val = self.q_value(state[:-1])
#         q_predict = batch_select(q_val, z)
#         assert q_predict.shape == q_target.shape
#         # assert torch.allclose(q_predict, batch_select(self.q_value(state), z))
#         return ((q_predict - q_target)**2).mean() # the $z$ selected should be consistent with the policy ..  




# class GaussianPolicy(PolicyZ):
#     def __init__(
#         self, state_dim, hidden_dim, z_space, cfg=None, K=1000000000, head=Normal.gdc(std_scale=1., std_mode='fix_no_grad', nocenter=True)
#     ) -> None:
#         super().__init__()
#         self.head = Normal(z_space, cfg=head)
#         self.qnet = self.build_backbone(state_dim, hidden_dim, self.head.get_input_dim())

#     def policy(self, state):
#         return self.head(self.qnet(state))

#     def loss(self, rollout):
#         with torch.no_grad():
#             value = rollout['value'][..., 0].detach()
#         logp = rollout['logp_z'][0].sum(axis=-1)
#         assert value.shape == logp.shape
#         pg = - logp * (value - value.mean()).detach()
#         return pg.mean()