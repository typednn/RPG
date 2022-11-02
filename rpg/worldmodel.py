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

        intrinsic_reward,

        cfg=None,
        gamma=0.99,
        lmbda=0.97,

        lmbda_last=False,


        horizon=1,
        markovian=False,
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

        self.intrinsic_reward = intrinsic_reward

        weights = lmbda_decay_weight(lmbda, horizon, lmbda_last=lmbda_last)
        self._weights = torch.nn.Parameter(torch.log(weights), requires_grad=True)

        self.markovian = markovian

    @property
    def weights(self):
        return torch.softmax(self._weights, 0)


    def policy(self, obs, z, timestep, return_z_dist=False):
        obs = totensor(obs, self.device)
        s = self.enc_s(obs, timestep=timestep)
        if self.pi_z is not None:
            if self.markovian:
                z_dist = self.pi_z(s, None, None, timestep)
            else:
                z = totensor(z, self.device, dtype=None)
                z_embed = self.enc_z(z)
                z_dist = self.pi_z(s, z, z_embed, timestep)
            if return_z_dist:
                return z_dist
            z =  z_dist.sample()[1]
        return self.pi_a(s, self.enc_z(z)), z

    def value(self, obs, z, timestep, detach=False):
        obs = totensor(obs, self.device)
        s = self.enc_s(obs, timestep=timestep)
        if detach:
            s = s.detach()
        if self.markovian:
            return self.value_fn(s)
        return self.value_fn(s, self.enc_z(z))

    def inference(self, obs, z, timestep, step, z_seq=None, a_seq=None, alpha=0., value_fn=None, pi_a=None, pi_z=None, pg=False):
        assert timestep.shape == (len(obs),)

        sample_z = (z_seq is None)
        if sample_z:
            z_seq, logp_z = [], []

        sample_a = (a_seq is None)
        if sample_a:
            a_seq, logp_a = [], []

        hidden = []
        states = []
        z_dones = []

        init_s = s = self.enc_s(obs, timestep=timestep)
        z_embed = self.enc_z(z)

        # for dynamics part ..
        h = self.init_h(s)[None,:]
        a_embeds = []

        if pi_a is None: pi_a = self.pi_a
        if pi_z is None: pi_z = self.pi_z

        #h = h.reshape(1, len(s), -1).permute(1, 0, 2).contiguous() # GRU of layer 2
        for idx in range(step):
            if len(z_seq) <= idx:
                if pi_z is not None:
                    z_done, z, _logp_z = pi_z(s, z, z_embed, timestep=timestep).sample()
                    z_dones.append(z_done)
                    logp_z.append(_logp_z)

                else:
                    logp_z.append(torch.zeros((len(s), 1), device='cuda:0', dtype=torch.float32))
                z_seq.append(z)
                # print(idx, z[0], s[0])
            z_embed = self.enc_z(z_seq[idx])

            if len(a_seq) <= idx:
                if pg:
                    a, _logp_a = pi_a(s, z_embed).sample()
                else:
                    a, _logp_a = pi_a(s, z_embed).rsample()
                logp_a.append(_logp_a[..., None])
                a_seq.append(a)

            a_embed = self.enc_a(a_seq[idx])
            a_embeds.append(a_embed)
            o, h = self.dynamic_fn(a_embed[None, :], h)
            assert torch.allclose(o[-1], h)
            s = self.state_dec(h[-1]) # predict the next hidden state ..

            hidden.append(h[-1])
            states.append(s)

            timestep = timestep + 1

        stack = torch.stack
        hidden = stack(hidden)
        states = stack(states)

        out = dict(hidden=hidden, states=states)
        if sample_a:
            out['a'] = stack(a_seq)
            logp_a = out['logp_a'] = stack(logp_a)
        if sample_z:
            z_seq = out['z'] = stack(z_seq)
            logp_z = out['logp_z'] = stack(logp_z)
            if len(z_dones) > 0:
                out['z_dones'] = stack(z_dones)

        #prefix = out['value_prefix'] = self.value_prefix(hidden)
        out['init_s'] = init_s
        out['rewards'] = rewards = self.value_prefix(states, stack(a_embeds)) # estimate rewards by the target states and the actions ..
        prefix = compute_value_prefix(rewards, self._cfg.gamma)


        if 'logp_a' in out:
            extra_rewards, infos = self.entropy_rewards(out['logp_a'], alpha)
            out.update(infos)

            #if hasattr(self, 'intrinsic_rewards'):
            if self.intrinsic_reward is not None:
                elbo, infos = self.intrinsic_reward.compute_reward(out)
                extra_rewards = extra_rewards + elbo
                # out.update(infos)
            
            if isinstance(extra_rewards, torch.Tensor):
                prefix = prefix + compute_value_prefix(extra_rewards, self._cfg.gamma)


        value_fn = (self.value_fn if value_fn is None else value_fn)
        if self.markovian:
            values = value_fn(states)
        else:
            values = value_fn(states, self.enc_z(z_seq))
            # raise NotImplementedError

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
        entropy_term = -logp_a.sum(axis=-1, keepdims=True)
        entropy = entropy_term * alpha if alpha > 0 else 0
        return entropy, {'entropy_term': entropy_term}
