from tools.utils import Seq, mlp, logger
import torch
from .utils import config_hidden
from .rpgm import Trainer

from nn.space import  Box, Discrete, MixtureSpace
# from nn.distributions.option import OptionNet
# from nn.distributions import CategoricalAction, MixtureAction, NormalAction, DistHead, ActionDistr
from nn.distributions import DistHead, ActionDistr
from nn.distributions.compositional import where

from .worldmodel import GeneralizedQ
from tools.utils import CatNet, orthogonal_init, TimedSeq

from tools.utils import logger
from tools.optim import LossOptimizer
from .models import BaseNet, Network

import torch

class Option(ActionDistr):
    def __init__(self, old, done_prob, new_distr, detach=True) -> None:
        super().__init__()

        self.old = old
        self.done_prob = done_prob
        self.new_distr = new_distr
        self.detach = detach

    def rsample(self, detach=False):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        d = torch.bernoulli(self.done_prob)
        if self.detach:
            a, logp_a = self.new_distr.sample()
        else:
            a, logp_a = self.new_distr.rsample()

        mask = d.bool()
        if not mask.all():
            logp_d = where(mask, torch.log(self.done_prob), torch.log(1 - self.done_prob))
            a = where(mask, a, self.old) # if done, use a
            logp_a = where(mask, logp_a, 0)
            # raise NotImplementedError
        else:
            logp_d = torch.zeros_like(logp_a)

        return d, a, torch.stack((logp_d, logp_a), dim=-1)

    def log_prob(self, action, sum=True):
        return self.new_distr.log_prob(action)


class OptionNet(Network):
    def __init__(self, zhead, backbone, backbone_dim, cfg=None, done_mode='samplefirst', detach=True) -> None:
        super().__init__()

        self.zhead = zhead
        self.backbone = backbone

        #inp = backbone.output_shape[-1]
        inp = backbone_dim
        self.option = torch.nn.Linear(inp, zhead.get_input_dim())
        self.done = torch.nn.Linear(inp, 1) # predict done probability

    def forward(self, s, z, z_embed, timestep):
        if self._cfg.done_mode == 'everystep':
            feature = self.backbone(s)
        else:
            feature = self.backbone(s, z_embed)

        if self._cfg.done_mode == 'samplefirst':
            done = (timestep == 0).float()
        elif self._cfg.done_mode == 'everystep':
            done = torch.ones_like(timestep).float()
        elif self._cfg.done_mode == 'option':
            done = torch.sigmoid(self.done(feature)[..., 0])
        else:
            raise NotImplementedError

        prob = self.option(feature)
        return Option(z, done, self.zhead(prob), detach=self._cfg.detach)


class IntrinsicReward(Network):
    # should use a transformer instead ..
    def __init__(
        self,
        state_dim, action_dim, hidden_dim, hidden_space,
        cfg=None,
        backbone=None,  action_weight=1., noise=0.0, obs_weight=1., head=None,
        entropy_coef=1.,
        mutual_info_weight=1.,
        use_next_state=False,
    ):
        super().__init__()

        zhead = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        backbone = Seq(mlp(state_dim + action_dim, hidden_dim, zhead.get_input_dim()))
        self.info_net = Seq(backbone, zhead)
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=True, device=self.device)
        )


        zhead2 = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        backbone = Seq(mlp(state_dim, hidden_dim, zhead2.get_input_dim()))
        self.posterior_z = Seq(backbone, zhead) # the posterior of p(z|s), for off-policy training..

    def get_alpha(self):
        return self.log_alpha.exp() * self._cfg.entropy_coef


    def forward(self, states, a_seq):
        states = states * self._cfg.obs_weight
        a_seq = (a_seq + torch.randn_like(a_seq) * self._cfg.noise)
        a_seq = a_seq * self._cfg.action_weight
        return self.info_net(states, a_seq)

    def get_state_seq(self, traj):
        if self._cfg.use_next_state:
            return traj['states']

        init_s = traj['init_s']
        states = traj['states']
        states = torch.cat((init_s[None,:], states[:-1]), dim=0)
        return states

    def compute_reward(self, traj):
        # states, a_seq, z_seq, logp_z, **kwargs
        states = self.get_state_seq(traj)

        a_seq = traj['a']
        z_seq = traj['z']
        logp_z = traj['logp_z']

        info =  self(states, a_seq).log_prob(z_seq) * self._cfg.mutual_info_weight # in case of discrete ..
        #alpha = self.log_alpha.exp().detach() * self._cfg.entropy_coef
        alpha = self.get_alpha()
        logger.logkvs_mean({'reward_info': float(info.mean()), 'z_alpha': float(alpha), 'reward_entz': float(-alpha * logp_z.sum(axis=-1).mean())})
        return (info - logp_z.sum(axis=-1) * alpha).unsqueeze(-1), {}

    def config_head(self, hidden_space):
        from tools.config import merge_inputs
        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=0.3989)

        if isinstance(hidden_space, Discrete):
            head = discrete
        elif isinstance(hidden_space, Box):
            head = continuous
        else:
            raise NotImplementedError
        if self._cfg.head is not None:
            head = merge_inputs(head, **self._cfg.head)
        return head


class OptionCritic(Trainer):
    def __init__(
        self, env,
        cfg=None,
        z_dim=10,
        ir=IntrinsicReward.dc,

        done_target=1./10, # control the entropy of done.

        entz_coef=1.,
        entz_target = None, # control the target entropies.
        option_mode='everystep',
        z_grad=False,
        optim_horizon = None,

        ppo=0,
    ):
        super().__init__(env)
        assert self.nets.intrinsic_reward is self.info_net
        self.info_net_optim = LossOptimizer(self.info_net, lr=3e-4) # info net

        from tools.utils import RunningMeanStd
        self.adv_norm = RunningMeanStd(clip_max=10.)

    def get_posterior(self, states):
        return self.info_net.posterior_z(states)

    def update_actor(self, obs, alpha, timesteps):
        t = timesteps[0]
        init_z = self.sample_z(obs, t).sample()[0]

        # rollout to get trajectories and values
        samples = self.nets.inference(obs, init_z, t, self._cfg.optim_horizon or self.horizon, alpha=alpha, pg=self._cfg.pg)
        value, entropy_term = samples['value'], samples['entropy_term']
        assert value.shape[-1] in [1, 2]
        estimated_value = value[..., 0]
        logp_z = samples['logp_z'][0].sum(axis=-1)

        with torch.no_grad():
            baseline = self.target_nets.value(obs, init_z, timestep=t)[..., 0]
            adv = (estimated_value -  baseline).detach()
            with torch.no_grad():
                self.adv_norm.update(adv.view(-1))
            adv = (adv - self.adv_norm.mean)/ self.adv_norm.std
            # adv = (adv - adv.mean(axis=0))/(adv.std(axis=0) + 1e-8) # normalize the advatnage ..

        if not self._cfg.z_grad:
            # mask = (logp_z < 0)
            # print(logp_z[mask], baseline[mask])
            if not self._cfg.ppo:
                assert adv.shape == logp_z.shape
                zdist = samples['init_z_dist'].new_distr
                from nn.distributions import NormalAction
                if isinstance(zdist, NormalAction) or True:
                    pi_z_loss = - (logp_z * adv).mean(axis=0) # - self.info_net.get_alpha() * (-logp_z).mean(axis=0) # not sure how to regularize the entropy ..
                else:
                    v = value[..., 0] / self.info_net.get_alpha() #TODO: instead of using the value, we can use the alpha to control the policy directly...
                    logits = samples['init_z_dist'].new_distr.logits
                    z0 = samples['z'][0]
                    weight = logits.gather(1, z0[:, None])[..., 0]
                    assert weight[2] == logits[2, z0[2]], (weight[2], logits[2, z0[2]])
                    pi_z_loss = ((weight - v)**2).mean()
            else:
                # prob ratio for KL / clipping based on a (possibly) recomputed logp
                assert self._cfg.option_mode == 'everystep'
                newlogp = logp_z
                with torch.no_grad():
                    import numpy as np
                    logp = self.old_pi.policy(obs, init_z, t, return_z_dist=True).log_prob(samples['z'][0]) # 1e-10 for numerical stability
                    logp = logp.clamp(np.log(1e-20), np.inf) # 1e-10 for numerical stability
                
                logratio = newlogp - logp
                ratio = torch.exp(logratio)
                # print(ratio.min(), torch.exp(logp).min(), adv.min(), adv.max())
                #if ratio.min() < 0.1:
                #    return {} # don't do the optimization ..
                # mask = torch.exp(logp) > 1e-10 # when it's zero, there should be no grad ..
                assert newlogp.shape == logp.shape
                assert adv.shape == ratio.shape, f"Adv shape is {adv.shape}, and ratio shape is {ratio.shape}"
                pg_losses = -adv * ratio
                clip_param = 0.2
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                pg_losses2 = -adv * clipped_ratio
                pg_losses = torch.max(pg_losses, pg_losses2)

                # pg_losses = (pg_losses * mask.float())
                pi_z_loss = pg_losses.mean(axis=0)
        else:
            pi_z_loss = 0. # joint optimize with a..


        # optimize pi_a
        if not self._cfg.pg:
            pi_a_loss = -estimated_value.mean(axis=0)
        else:
            logp_a = samples['logp_a'][0].sum(axis=-1)
            assert logp_a.shape == adv.shape
            pi_a_loss = - (logp_a * adv).mean(axis=0) - alpha * (-logp_a).mean(axis=0) / self.adv_norm.std
            raise NotImplementedError

        self.actor_optim.optimize(pi_a_loss + pi_z_loss)
        if self._cfg.entropy_target is not None:
            entropy_loss = -torch.mean(
                self.log_alpha.exp() * (self._cfg.entropy_target - entropy_term.detach()))
            self.entropy_optim.optimize(entropy_loss)

        # optimize auxilary losses 
        s_seq = self.info_net.get_state_seq(samples).detach()
        z_detach = samples['z'].detach()

        dist = self.info_net(s_seq, samples['a'].detach())
        mutual_info = dist.log_prob(z_detach)
        #print(samples['a'][0][10], self.info_net(s_seq, samples['a'].detach()).dist.mean[0, 10], z_detach[0, 10], mutual_info[0, 10])
        #print(dist.dist.scale[0, 10])
        # import numpy as np
        # def log_prob(z, loc, scale):
        #     return -0.5 * np.log(2 * np.pi) - torch.log(scale) - 0.5 * ((z - loc) / scale) ** 2
        # print(np.exp(0.5 * np.log(2*np.pi)))
        # exit(0)
        mutual_info = mutual_info.mean()
        posterior = self.get_posterior(samples['states'].detach()).log_prob(z_detach).mean()

        z_entropy = -logp_z.mean()
        if self._cfg.entz_target is not None:
            z_entropy_loss = -torch.mean(
                self.info_net.log_alpha.exp() * (
                    self._cfg.entz_target - z_entropy.detach())
            )
        else:
            z_entropy_loss = 0
        self.info_net_optim.optimize(z_entropy_loss - mutual_info - posterior)


        return {
            'a_entropy': float(entropy_term.mean()),
            'a_pi_loss': float(pi_a_loss),
            'z_pi_loss': float(pi_z_loss),
            'adv_abs': float(adv.abs().mean()),
            'z_alpha_loss': float(z_entropy_loss),
            'z_entropy': float(z_entropy),
            'z_posterior': float(posterior),
            'info_loss': float(mutual_info),
        }

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        latent_dim = 100
        option_mode = self._cfg.option_mode

        z_dim = z_space.inp_shape[0]
        latent_z_dim = z_dim
        action_dim = action_space.shape[0]

        # TODO: layer norm?
        from .utils import ZTransform, config_hidden
        enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim))
        enc_z = ZTransform(z_space)
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
        # dynamics = MLPDynamics(hidden_dim)

        value_prefix = Seq(mlp(latent_dim + hidden_dim, hidden_dim, 1))
        done_fn = mlp(hidden_dim, hidden_dim, 1) if self._cfg.have_done else None
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..


        v_in = latent_dim + latent_z_dim if option_mode != 'everystep' else latent_dim
        value = CatNet(
            Seq(mlp(v_in, hidden_dim, 1)),
            Seq(mlp(v_in, hidden_dim, 1)),
        ) #layer norm, if necesssary
        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(latent_dim + latent_z_dim, hidden_dim, head.get_input_dim()), head)


        zhead = DistHead.build(z_space, cfg=config_hidden(self._cfg.z_head, z_space))
        backbone = Seq(mlp(v_in, hidden_dim, hidden_dim))
        pi_z = OptionNet(zhead, backbone, hidden_dim, done_mode=self._cfg.option_mode, detach=not self._cfg.z_grad)


        self.info_net = IntrinsicReward(latent_dim, action_dim, hidden_dim, self.z_space, entropy_coef=self._cfg.entz_coef, cfg=self._cfg.ir)

        network = GeneralizedQ(
            enc_s, enc_a, enc_z, pi_a, pi_z, init_h, dynamics, state_dec, value_prefix, value, done_fn,
            self.info_net,

            cfg=self._cfg.qnet,
            horizon=self._cfg.horizon,
            markovian= (option_mode == 'everystep'),
        )
        network.apply(orthogonal_init)
        return network
