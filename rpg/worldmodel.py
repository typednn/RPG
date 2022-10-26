"""
# model-based verison RPG
s      a_1    a_2    ...
|      |      |
h_0 -> h_1 -> h_2 -> h_3 -> h_4
        |      |      |      |   
        o_1    o_2    o_3    o_4
        / |  
    s1 r1    
    |
    v1
"""
import torch
from tools.nn_base import Network
from tools.utils import totensor
from .utils import done_rewards_values, lmbda_decay_weight, compute_value_prefix


class GeneralizedQ(Network):
    def __init__(
        self,
        enc_s, enc_a, enc_z,
        pi_a, pi_z,

        init_h, dynamic_fn,
        state_dec, value_prefix, value_fn,

        done_fn,

        cfg=None,
        gamma=0.99,
        lmbda=0.97,

        lmbda_last=False,


        horizon=1,
    ) -> None:
        super().__init__()

        self.pi_a, self.pi_z = pi_a, pi_z
        self.enc_s, self.enc_z, self.enc_a = enc_s, enc_z, enc_a

        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn

        self.state_dec = state_dec
        self.value_prefix = value_prefix
        self.value_fn = value_fn

        self.done_fn = done_fn
            

        v_nets = [enc_s, enc_z, value_fn]
        d_nets = [enc_a, init_h, dynamic_fn, state_dec, value_prefix]

        self.dynamics = torch.nn.ModuleList(d_nets + v_nets)
        self.policies = torch.nn.ModuleList([pi_a, pi_z])

        weights = lmbda_decay_weight(lmbda, horizon, lmbda_last=lmbda_last)
        self._weights = torch.nn.Parameter(torch.log(weights), requires_grad=True)

    @property
    def weights(self):
        return torch.softmax(self._weights, 0)


    def policy(self, obs, z):
        obs = totensor(obs, self.device)
        z = totensor(z, self.device, dtype=None)
        z_embed = self.enc_z(z)
        s = self.enc_s(obs)
        z = self.pi_z(s, z_embed).sample()[0]
        return self.pi_a(s, self.enc_z(z)), z

    def value(self, obs, z):
        obs = totensor(obs, self.device)
        s = self.enc_s(obs)
        return self.value_fn(s, self.enc_z(z))

    def inference(self, obs, z, step, z_seq=None, a_seq=None, alpha=0., value_fn=None):
        sample_z = (z_seq is None)
        if sample_z:
            z_seq, logp_z = [], []
        sample_a = (a_seq is None)
        if sample_a:
            a_seq, logp_a = [], []

        hidden = []
        states = []

        s = self.enc_s(obs)
        z_embed = self.enc_z(z)

        # for dynamics part ..
        h = self.init_h(s)[None,:]
        #h = h.reshape(1, len(s), -1).permute(1, 0, 2).contiguous() # GRU of layer 2
        for idx in range(step):
            if len(z_seq) <= idx:
                z, logp = self.pi_z(s, z_embed).sample()
                logp_z.append(logp[..., None])
                z_seq.append(z)
            z_embed = self.enc_z(z_seq[idx])

            if len(a_seq) <= idx:
                a, logp = self.pi_a(s, z_embed).rsample()
                logp_a.append(logp[..., None])
                a_seq.append(a)

            a_embed = self.enc_a(a_seq[idx])
            o, h = self.dynamic_fn(a_embed[None, :], h)
            assert torch.allclose(o[-1], h)
            s = self.state_dec(h[-1]) # predict the next hidden state ..

            hidden.append(h[-1])
            states.append(s)

        stack = torch.stack
        hidden = stack(hidden)
        states = stack(states)

        out = dict(hidden=hidden, states=states)
        if sample_a:
            out['a'] = stack(a_seq)
            out['logp_a'] = stack(logp_a)
        if sample_z:
            z_seq = out['z'] = stack(z_seq)
            out['logp_z'] = stack(logp_z)
            assert (z_seq == 0).all()
        prefix = out['value_prefix'] = self.value_prefix(hidden)

        if 'logp_a' in out:
            extra_rewards, infos = self.entropy_rewards(out['logp_a'], alpha)
            out.update(infos)
            prefix = prefix + compute_value_prefix(extra_rewards, self._cfg.gamma)


        values = (self.value_fn if value_fn is None else value_fn)(states, self.enc_z(z_seq))

        out['dones'] = dones = torch.sigmoid(self.done_fn(hidden)) if self.done_fn is not None else None
        expected_values, expected_prefix = done_rewards_values(values, prefix, dones)

        gamma = 1
        vpreds = []
        for i in range(len(hidden)):
            vpred = (expected_prefix[i] + expected_values[i] * gamma * self._cfg.gamma)
            vpreds.append(vpred)
            gamma *= self._cfg.gamma

        vpreds = stack(vpreds)
        out['value'] = (vpreds * self.weights[:, None, None]).sum(axis=0)
        out['next_values'] = values
        return out


    def entropy_rewards(self, logp_a, alpha=0.):
        entropy_term = -logp_a
        entropy = entropy_term * alpha if alpha > 0 else 0
        return entropy, {'entropy_term': entropy_term}
