# model-based verison RPG
import torch
from tools.config import Configurable
from tools.nn_base import Network
from .utils import compute_gae_by_hand

def compute_value_prefix(rewards, gamma):
    value_prefix = [None] * len(rewards)
    v = 0
    for i in range(len(rewards) - 1, -1, 0):
        v = v * gamma + rewards[i]
        value_prefix.append(v)
    return torch.stack(value_prefix)


class GeneralizedQ(Network):
    # predict state from the hidden.
    # for simplicity, we assume there is no done,
    def __init__(
        self,
        enc_s, enc_a, enc_z,
        pi_a, pi_z,

        init_h, dynamics_fn,
        state_dec, value_prefix, value_fn,

        cfg=None,
        gamma=0.995,
        lmbda=0.97,
        rollout_step = 10,
    ) -> None:
        super().__init__()

        self.enc_s = enc_s
        self.enc_z = enc_z
        self.enc_a = enc_a

        self.pi_a = pi_a
        self.pi_z = pi_z
        assert self.pi_z is None

        self.init_h = init_h
        self.dynamic_fn = dynamics_fn # RNN ..

        self.state_dec = state_dec
        self.value_prefix = value_prefix
        self.value_fn = value_fn

    def rollout(self, obs, seq_a):
        h = self.dynamic_fn.init_state(self.enc_s(obs))
        hidden = []
        for a_ in self.enc_a(seq_a):
            h = self.dynamic_fn(h, a_) # should be RNN
            hidden.append(h)
        return torch.stack(hidden)

    def policy(self, obs, z):
        # sample $a$ directly from the policy ..
        #return self.sample(obs, z, 1)['actions'][0]
        return self.pi_a(self.enc_s(obs), self.enc_z(z))

    def sample(self, obs, z, step):
        # we make the following assumption
        # during the model-based rollout, we will never sample $z$

        z_embed = self.enc_z(z)
        s = self.enc_s(obs)
        h = self.init_h(s)
        s_recon = self.state_dec(h)

        hidden, logp_a, actions, z_embeds, states = [], [], [], [], [s_recon]
        for _ in range(step):
            assert self.pi_z is None, "this means that we will not update z, anyway.."
            a, logp = self.pi_a(s, z_embed).rsample()
            h = self.dynamic_fn(h, a)
            s = self.state_dec(h) # predict the next hidden state ..
            hidden.append(h); actions.append(a); logp_a.append(logp); z_embeds.append(z_embed);
            states.append(s)
        return {
            'hidden': torch.stack(hidden),
            'actions': torch.stack(actions),
            'logp_a': torch.stack(logp_a),
            'z_embed': torch.stack(z_embed), 
            'states': torch.stack(states),
            # logp_z and mutual info is None right now ..
        }

    # rewards means other rewards ..
    def predict_values(self, hidden, z_embed, rewards=None):
        hidden = torch.stack(hidden)


        value_prefix = self.value_prefix(hidden) # predict reward

        if rewards is not None:
            value_prefix = value_prefix + compute_value_prefix(rewards)

        values = self.value_fn(hidden, z_embed) # predict V(s, z_{t-1})
        # states = self.state_dec(hidden)

        gamma = 1
        lmbda = 1
        sum_lmbda = 0.
        v = 0
        for i in range(len(hidden)):
            vpred = (value_prefix[i] + values[i] * gamma * self._cfg.gamma)
            v = v + vpred * lmbda
            sum_lmbda += lmbda
            gamma *= self._cfg.gamma
            lmbda *= self._cfg.lmbda
        v = (v + (1./(1-self._cfg.lmbda) - sum_lmbda) * vpred) * (1-self._cfg.lmbda)
        return torch.cat((v[None,:], values)), value_prefix

    def estimate_value(self, obs, z, alpha=0):
        traj = self.sample(obs, z, self._cfg.rollout_step)
        extra_reward = traj['logp_a'] * alpha
        return self.predict_values(traj['hidden'], traj['z_embed'], extra_reward) 

    @torch.no_grad()
    def compute_loss(self, obs, rewards, target_values):
        # obs: T+1
        # actions, rewards: T
        # assume that we have some way to estimate next_values ..
        states = self.enc_s(obs)
        value_prefix = compute_value_prefix(rewards, self._cfg.gamma)

        # gae ..
        from .utils import compute_gae_by_hand
        zero = torch.zeros_like(target_values)
        vtarg = compute_gae_by_hand(
            rewards, zero, target_values, zero, zero,
            self._cfg.gamma, self._cfg.lmbda, mode='exact'
        )
        return states, value_prefix, vtarg


class IgnoreNN(torch.nn.Module):
    def __init__(self, main):
        super().__init__()
        self.main = main
    def forward(self, x, z=None):
        if z is not None: x = torch.cat([x, z], dim=-1)
        return self.main(x)

class Identity(torch.nn.Module):
    def forward(self, x):
        return x

class train_model_based(Configurable):
    def __init__(self, cfg=None, nsteps=2000):
        super().__init__(cfg)

    def make_network(self, obs_space, action_space, z_space=None):
        from tools.utils import mlp, orthogonal_init
        dim = 256
        enc_s = mlp(obs_space.shape[0], dim, dim)
        enc_z = None
        enc_a = mlp(action_space.shape[0], dim, dim)
        dynamics = mlp(dim + dim, dim, dim)
        state_dec = mlp(obs_space.shape[0], dim, dim) # reconstruct obs ..
        init_h = mlp(dim, dim, dim) # reconstruct obs ..

        value = IgnoreNN(mlp(dim, dim, 1))
        value_prefix = IgnoreNN(mlp(dim, dim, 1))

        
    def train(self):
        # every time sample a batch of data
        # sample indexes from where we need to sample new z (or not, but do not across where we sample z ..)
        # Iterative:
        #   for optimizing a:
        #       optimize the low-level value and dynamics model ..
        #   for optimizing z:
        #       directly optimize it with the value network for any states ..

        # the objective is that: the policy is optimized separately
        # the dynamics should ensures the recons (hidden) and predict future states  
        # 