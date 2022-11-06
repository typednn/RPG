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
from tools.utils import totensor
from tools.nn_base import Network

from .utils import lmbda_decay_weight


class GeneralizedQ(Network):
    def __init__(
        self,
        enc_s, enc_a, pi_a, pi_z,
        init_h, dynamic_fn, state_dec, reward_predictor, q_fn,
        done_fn,
        intrinsic_reward,
        gamma=0.99, lmbda=0.97, lmbda_last=False, horizon=1,
    ) -> None:
        super().__init__()
        self.gamma = gamma

        self.pi_a, self.pi_z = pi_a, pi_z
        self.enc_s, self.enc_a = enc_s, enc_a

        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn

        self.state_dec = state_dec
        self.reward_predictor = reward_predictor
        self.q_fn = q_fn

        self.done_fn = done_fn
            
        self.dynamics = torch.nn.ModuleList([enc_s, q_fn, enc_a, init_h, dynamic_fn, state_dec])
        self.policies = torch.nn.ModuleList([pi_a, pi_z])

        self.intrinsic_reward = intrinsic_reward

        weights = lmbda_decay_weight(lmbda, horizon, lmbda_last=lmbda_last)
        self._weights = torch.nn.Parameter(torch.log(weights), requires_grad=True)

    def set_alpha(self, alpha_a, alpha_z):
        alpha_a = float(alpha_a)
        alpha_z = float(alpha_z)
        # alpha = torch.concat((alpha_a, ))
        alpha = torch.tensor([alpha_a, alpha_z], self.device)
        self.alpha = alpha

        self.pi_a.set_alpha(alpha)
        self.pi_z.set_alpha(alpha)
        self.q_fn.set_alpha(alpha)

    @property
    def weights(self):
        return torch.softmax(self._weights, 0)


    def policy(self, obs, prevz, timestep):
        obs = totensor(obs, self.device)
        prevz = totensor(prevz, self.device, dtype=None)
        s = self.enc_s(obs, timestep=timestep)
        z = self.pi_z(s, prevz, timestep).z
        return self.pi_a(s, z).a, z

    def value(self, obs, prevz, timestep, z, a, detach=False):
        obs = totensor(obs, self.device)
        prevz = totensor(prevz, self.device, dtype=None)
        a = totensor(a, self.device)
        z = totensor(z, self.device)

        s = self.enc_s(obs, timestep=timestep)
        if detach:
            s = s.detach()
        return self.value_fn(s, z, a)


    def inference(
        self, obs, z, timestep, step, z_seq=None, a_seq=None, pi_a=None, pi_z=None, pg=False):
        assert timestep.shape == (len(obs),)

        sample_z = (z_seq is None)
        if sample_z:
            z_seq, logp_z, entz = [], [], []

        sample_a = (a_seq is None)
        if sample_a:
            a_seq, logp_a = [], []

        hidden = []

        s = self.enc_s(obs, timestep=timestep)
        states = [s]

        # for dynamics part ..
        h = self.init_h(s)[None,:]
        a_embeds = []

        if pi_a is None: pi_a = self.pi_a
        if pi_z is None: pi_z = self.pi_z

        for idx in range(step):
            if len(z_seq) <= idx:
                if pi_z is not None:
                    z, _logp_z, z_done, logp_z_done, _entz = pi_z(s, z, timestep=timestep)
                    logp_z.append(_logp_z[..., None])
                    entz.append(_entz[..., None])
                else:
                    logp_z.append(torch.zeros((len(s), 1), device='cuda:0', dtype=torch.float32))
                z_seq.append(z)
            z = z_seq[idx]

            if len(a_seq) <= idx:
                a, _logp_a = pi_a(s, z).sample()
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
        out = dict(hidden=hidden, state=states, reward=self.reward_predictor(states[1:], stack(a_embeds)))

        if sample_a:
            out['a'] = stack(a_seq)
            out['logp_a'] = stack(logp_a)

        if sample_z:
            out['z'] = stack(z_seq)
            out['logp_z'] = stack(logp_z)
            out['ent_z'] = stack(entz)

        if sample_z:
            rewards, entropies, infos = self.intrinsic_reward.estimate_unscaled_rewards(out) # return rewards, (ent_a,ent_z)
            out.update(infos)
            q_values = self.q_fn(states[:-1], out['z'], out['a'], states[1:]) 
            out['done'] = dones = torch.sigmoid(self.done_fn(hidden)) if self.done_fn is not None else None

            discount = 1

            prefix = 0.
            vpreds = []
            for i in range(len(hidden)):
                # add the entropies at the current steps ..
                vpred = (prefix + (entropies[i].sum(axis=-1, keepdims=True) + q_values[i]) * discount)
                vpreds.append(vpred)

                prefix = prefix + rewards[i] * discount

                discount *= self.gamma
                if dones is not None:
                    assert dones.shape[-1] == 1
                    discount = discount * (1 - dones[i]) # the probablity of not done ..

            vpreds = stack(vpreds)
            out['value'] = (vpreds * self.weights[:, None, None, None]).sum(axis=0)
            out['q_value'] = q_values
            out['entropy'] = entropies

        assert out['value'].shape[-1] == 2, "must be double q learning .."
        # TODO: add total_value
        return out