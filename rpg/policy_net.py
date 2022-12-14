import torch
from collections import namedtuple
from tools.utils import logger
from .critic import AlphaPolicyBase
from nn.distributions import NormalAction

def batch_select(values, z=None):
    if z is None:
        return values
    else:
        return torch.gather(values, -1, z.unsqueeze(-1))


Aout = namedtuple('Aout', ['a', 'logp', 'ent'])

class PolicyBase(AlphaPolicyBase):
    def __init__(self, state_dim, hidden_dim, head, cfg=None):
        super().__init__()
        self.head = head
        self.backbone = self.build_backbone(
            state_dim,
            hidden_dim,
            head.get_input_dim()
        )

    def loss(self, rollout, alpha):
        return self.head.loss(rollout, alpha)

    def forward(self, inp, alpha):
        feature = self.backbone(inp)
        dist =  self.head(feature)


        if isinstance(dist, NormalAction):
            scale = dist.dist.scale
            logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})

        a, logp = dist.rsample()

        return Aout(a, logp, -logp)


class DiffPolicy(PolicyBase):
    def __init__(self, state_dim, hidden_dim, head, cfg=None):
        super().__init__(state_dim, hidden_dim, head)

    def loss(self, rollout, alpha):
        assert rollout['value'].shape[-1] == 2
        return -rollout['value'][..., 0].mean()



class QPolicy(PolicyBase):
    def __init__(
        self, state_dim, hidden_dim, head, cfg=None,
    ) -> None:
        super().__init__(state_dim, hidden_dim, head)
        #self.qnet = self.build_backbone(state_dim, hidden_dim, head.get_input_dim())

    def q_value(self, state):
        return self.backbone(state)

    def forward(self, state, alpha):
        q = self.q_value(state)
        logits = q / alpha
        out = self.head(logits)
        a, logp = out.sample()
        return Aout(a, logp, out.entropy())

    def loss(self, rollout, alpha):
        # for simplicity, we directly let the high-level policy to select the action with the best z values .. 
        state = rollout['state'].detach()
        q_value = rollout['q_value'].detach()
        z = rollout['z'].detach()

        q_value = q_value.min(axis=-1, keepdims=True).values
        with torch.no_grad():
            q_target = q_value
            extra_rewards = rollout['extra_rewards']
            for k, v in extra_rewards.items():
                if k!='ent_z':
                    q_target = q_target + v.detach()

        q_val = self.q_value(state[:-1])
        q_predict = batch_select(q_val, z)
        assert q_predict.shape == q_target.shape
        return ((q_predict - q_target)**2).mean() # the $z$ selected should be consistent with the policy ..  