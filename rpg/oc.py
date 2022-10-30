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
    def __init__(self, old, done_prob, new_distr) -> None:
        super().__init__()

        self.old = old
        self.done_prob = done_prob
        self.new_distr = new_distr

    def rsample(self, detach=False):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        d = torch.bernoulli(self.done_prob)
        a, logp_a = self.new_distr.sample()

        mask = d.bool()
        logp_d = where(mask, torch.log(self.done_prob), torch.log(1 - self.done_prob))
        a = where(mask, a, self.old) # if done, use a
        logp_a = where(mask, logp_a, 0)
        return d, a, torch.stack((logp_d, logp_a), dim=-1)

class OptionNet(Network):
    def __init__(self, zhead, backbone, backbone_dim, cfg=None, done_mode='sample_first') -> None:
        super().__init__()

        self.zhead = zhead
        self.backbone = backbone

        #inp = backbone.output_shape[-1]
        inp = backbone_dim
        self.option = torch.nn.Linear(inp, zhead.get_input_dim())
        self.done = torch.nn.Linear(inp, 1) # predict done probability

    def forward(self, s, z, z_embed, timestep):
        feature = self.backbone(s, z_embed)
        if self._cfg.done_mode == 'sample_first':
            done = (timestep == 0).float()
        else:
            done = torch.sigmoid(self.done(feature)[..., 0])

        prob = self.option(feature)
        return Option(z, done, self.zhead(prob))


class IntrinsicReward(Network):
    # should use a transformer instead ..
    def __init__(
        self,
        state_dim, action_dim, hidden_dim, hidden_space,
        cfg=None,
        backbone=None,  action_weight=1., noise=0.0, obs_weight=1., head=None,
        entropy_coef=1.,
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

    def forward(self, states, a_seq):
        states = states * self._cfg.obs_weight
        a_seq = (a_seq + torch.randn_like(a_seq) * self._cfg.noise)
        a_seq = a_seq * self._cfg.action_weight
        return self.info_net(states, a_seq)

    def compute_reward(self, traj):
        # states, a_seq, z_seq, logp_z, **kwargs
        states = traj['states']
        a_seq = traj['a']
        z_seq = traj['z']
        logp_z = traj['logp_z']

        info =  self(states, a_seq).log_prob(z_seq) # in case of discrete ..
        alpha = self.log_alpha.exp().detach() * self._cfg.entropy_coef
        return (info - logp_z.sum(axis=-1) * alpha).unsqueeze(-1), {
            'info_reward': float(info.mean())
        }

    def config_head(self, hidden_space):
        from tools.config import merge_inputs
        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=1.)

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
    ):
        super().__init__(env)
        self.info_net_optim = LossOptimizer(self.nets.intrinsic_reward, lr=3e-4) # info net

    def get_posterior(self, states):
        return self.info_net.posterior_z(states)

    def update_actor(self, obs, alpha, timesteps):
        t = timesteps[0]
        init_z = self.sample_z(obs, t).sample()[0]

        # rollout to get trajectories and values
        samples = self.nets.inference(obs, init_z, t, self.horizon, alpha=alpha, pg=self._cfg.pg)
        value, entropy_term = samples['value'], samples['entropy_term']
        assert value.shape[-1] in [1, 2]
        estimated_value = value[..., 0]

        # optimize pi_z with policy gradient directly
        baseline = self.nets.value(obs, init_z, timestep=t)[..., 0]
        logp_z = samples['logp_z'][0].sum(axis=-1)

        adv = (estimated_value -  baseline).detach()
        adv = (adv - adv.mean(axis=0))/(adv.std(axis=0) + 1e-8) # normalize the advatnage ..
        assert adv.shape == logp_z.shape
        pi_z_loss = - (logp_z * adv).mean(axis=0) # not sure how to regularize the entropy ..


        # optimize pi_a
        if not self._cfg.pg:
            pi_a_loss = -estimated_value.mean(axis=0)
        else:
            logp_a = samples['logp_a'][0].sum(axis=-1)
            assert logp_a.shape == adv.shape
            pi_a_loss = - (logp_a * adv).mean(axis=0) - alpha * (-logp_a).mean(axis=0)
            raise NotImplementedError

        self.actor_optim.optimize(pi_a_loss + pi_z_loss)
        if self._cfg.entropy_target is not None:
            entropy_loss = -torch.mean(
                self.log_alpha.exp() * (self._cfg.entropy_target - entropy_term.detach()))
            self.entropy_optim.optimize(entropy_loss)

        # optimize auxilary losses 
        mutual_info = self.info_net(
            samples['states'].detach(), samples['a'].detach()).log_prob(samples['z'].detach()).mean()
        posterior = self.get_posterior(samples['states'].detach()).log_prob(samples['z'].detach()).mean()

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
            'z_alpha_loss': float(z_entropy_loss),
            'z_entropy': float(z_entropy),
            'z_posterior': float(posterior),
            'info_loss': float(mutual_info),
        }

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        latent_dim = 100

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


        v_in = latent_dim + latent_z_dim
        value = CatNet(
            Seq(mlp(v_in, hidden_dim, 1)),
            Seq(mlp(v_in, hidden_dim, 1)),
        ) #layer norm, if necesssary
        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(latent_dim + latent_z_dim, hidden_dim, head.get_input_dim()), head)


        zhead = DistHead.build(z_space, cfg=config_hidden(self._cfg.z_head, z_space))
        backbone = Seq(mlp(latent_dim + latent_z_dim, hidden_dim, hidden_dim))
        pi_z = OptionNet(zhead, backbone, hidden_dim)


        self.info_net = IntrinsicReward(latent_dim, action_dim, hidden_dim, self.z_space, entropy_coef=self._cfg.entz_coef)

        network = GeneralizedQ(
            enc_s, enc_a, enc_z, pi_a, pi_z, init_h, dynamics, state_dec, value_prefix, value, done_fn,
            self.info_net,

            cfg=self._cfg.qnet,
            horizon=self._cfg.horizon
        )
        network.apply(orthogonal_init)
        return network