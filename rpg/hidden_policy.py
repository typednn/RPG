import torch
from nn.distributions import Normal
from tools.nn_base import Network
from tools.config import as_builder
from tools.utils import Seq, mlp 
from .soft_actor_critic import AlphaPolicyBase, Zout, batch_select


@as_builder
class HiddenPolicy(Network):
    def __init__(self, state_dim, z_space, cfg=None):
        super().__init__(cfg)
        self.state_dim = state_dim
        self.z_space = z_space

        
class GaussianPolicy(HiddenPolicy):
    def __init__(self, state_dim, z_space, cfg=None, head=Normal.gdc(std_scale=1., nocenter=True, std_mode='fix_no_grad')):
        super().__init__(state_dim, z_space, cfg)
        self.head = Normal(z_space, cfg=head)
        self.backbone = Seq(mlp(state_dim, 256, self.head.get_input_dim()))

    def forward(self, state):
        return self.backbone(self.head(state))

    def loss(self, rollout):
        # logp_z = rollout['logp_z'][0]
        # value = rollout['value'][0]
        # # baseline ??
        # return  -(logp_z * value.detach()).mean() # policy gradient ..
        raise NotImplementedError
        return self.policy.loss(rollout)


class SoftPolicy(HiddenPolicy):
    def __init__(self, state_dim, z_space, cfg=None, epsilon=0.01):
        super().__init__(state_dim, z_space, cfg)
        self.qnet = mlp(state_dim, 256, z_space.n)

    def q_value(self, state):
        return self.qnet(self.add_alpha(state))

    def forward(self, state, prevz, timestep, z=None):
        q = self.q_value(state)
        logits = q / self.alpha[1]
        if self._cfg.epsilon  == 0.:
            pi_z = torch.distributions.Categorical(logits=logits) # the second is the z..
        else:
            prob = torch.softmax(logits, dim=-1) * (1 - self._cfg.epsilon) + self._cfg.epsilon / logits.shape[-1]
            pi_z = torch.distributions.Categorical(probs=prob)
        return pi_z


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



class PolicyZ(AlphaPolicyBase):
    def __init__(self, policy, cfg=None, K=1000000000) -> None:
        super().__init__()
        self.policy = policy


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

    def loss(self, rollout):
        return self.policy.loss(rollout)