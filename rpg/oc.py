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
        d = torch.bernoulli(self.done_prob)
        logp_d = torch.log(self.done_prob) * d + torch.log(1 - self.done_prob) * (1 - d)
        a, logp_a = self.new_distr.rsample(detach)

        a = where(d, a, self.old) # if done, use a
        logp_a = where(d, logp_a, 0)
        d = d.float()
        return d, a, torch.stack((logp_d, logp_a), dim=-1)

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        return self.rsample(detach=True)

class OptionNet(Network):
    def __init__(self, zhead, backbone, cfg=None, done_mode='sample_first') -> None:
        super().__init__()

        self.zhead = zhead
        self.backbone = backbone

        inp = backbone.output_shape[-1]
        self.option = torch.nn.Linear(inp, zhead.get_input_dim())
        self.done = torch.nn.Linear(inp, 1) # predict done probability

    def forward(self, s, z, timestep):
        feature = self.backbone(s, z)
        if self._cfg.done_mode == 'sample_first':
            done = (timestep == 0).float()
        else:
            done = torch.sigmoid(self.done(feature))
        return Option(z, done, self.zhead(self.option(feature)))

class IntrinsicReward(Network):
    # should use a transformer instead ..
    def __init__(
        self,
        obs_space,
        action_space,
        hidden_space,
        cfg=None,
        backbone=None,

        # parameters for infonet ..
        action_weight=1.,
        noise=0.0,
        obs_weight=1.,
        head=None,

        entropy_coef=1.,
    ):
        super().__init__()
        self.info_net = BaseNet(
            (obs_space, action_space),
            hidden_space,
            backbone=backbone,
            head=self.config_head(hidden_space)
        ).cuda()
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=True, device=self.device)
        )

    def forward(self, states, a_seq):
        states = states * self._cfg.obs_weight
        a_seq = (a_seq + torch.randn_like(a_seq) * self._cfg.noise)
        a_seq = a_seq * self._cfg.action_weight
        return self.info_net(states, a_seq)

    def compute(self, states, a_seq, z_seq, logp_z, **kwargs):
        info =  self(states, a_seq).log_prob(z_seq)
        alpha = self.log_alpha.exp().detach() * self._cfg.entropy_coef
        return info - logp_z[..., 1:] * alpha

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
        ir=IntrinsicReward.to_build(TYPE=IntrinsicReward),

        done_target=1./10, # control the entropy of done.
        entz_target = None, # control the target entropies.
    ):
        super().__init__(env)
        self.info_net_optim = LossOptimizer(self.nets.intrinsic_reward, lr=3e-4) # info net

    def sample_hidden_z(self, obs, timestep, action, next_obs):
        s = self.net.enc_s(next_obs[0], timestep + 1)
        return self.info_net.net(s, action).sample()

    def update_actor(self, obs, init_z, alpha, timesteps):
        # optimize pi_a
        samples = self.nets.inference(obs, init_z, timesteps, self.horizon, alpha=alpha)
        value, entropy_term = samples['value'], samples['entropy_term']
        assert value.shape[-1] in [1, 2]

        estimated_value = value[..., 0]
        pi_a_loss = - estimated_value.mean(axis=0)

        # optimize pi_z with policy gradient directly
        baseline = self.nets.value(obs, init_z)
        logp_z = samples['logp_z'][0].sum(axis=-1)

        adv = (estimated_value -  baseline).detach()
        assert adv.shape[-1] == logp_z.shape[-1]
        pi_z_loss = - (logp_z * adv).mean(axis=0)

        self.actor_optim.optimize(pi_a_loss + pi_z_loss)
        if self._cfg.entropy_target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self._cfg.entropy_target - entropy_term.detach()))
            self.entropy_optim.optimize(entropy_loss)

        # optimize the elbo loss and the z entropy
        #sampled_z = samples['z_seq'][0]
        mutual_info = self.info_net(
            samples['states'], samples['a_seq']).log_prob(samples['z_seq']).mean(axis=0)
        if self._cfg.entz_target is not None:
            z_entropy = -logp_z
            z_entropy_loss = -torch.mean(
                self.info_net.log_alpha.exp() * (
                    self._cfg.entz_target - z_entropy.detach())
            )
        else:
            z_entropy_loss = 0
        self.info_net_optim.optimize(z_entropy_loss + mutual_info)


        return {
            'pi_a_loss': float(pi_a_loss),
            'pi_z_loss': float(pi_z_loss),
            'entropy': float(entropy_term.mean()),
        }


    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        latent_dim = 100

        z_dim = z_space.inp_shape[0]
        latent_z_dim = z_dim

        # TODO: layer norm?
        from .utils import ZTransform, config_hidden
        enc_s = mlp(obs_space.shape[0], hidden_dim, latent_dim) 
        enc_z = ZTransform(z_space)
        enc_a = TimedSeq(mlp(action_space.shape[0], hidden_dim, hidden_dim))

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
        # dynamics = MLPDynamics(hidden_dim)

        value_prefix = Seq(mlp(hidden_dim, hidden_dim, 1))
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
        backbone = mlp(latent_dim + latent_z_dim, hidden_dim, hidden_dim)
        pi_z = OptionNet(zhead, backbone)


        self.info_net = IntrinsicReward.build(
            obs_space, action_space, self.z_space, cfg=self._cfg.ir)

        network = GeneralizedQ(
            enc_s, enc_a, enc_z, pi_a, pi_z, init_h, dynamics, state_dec, value_prefix, value, done_fn,
            self.info_net,

            cfg=self._cfg.qnet,
            horizon=self._cfg.horizon
        )
        network.apply(orthogonal_init)
        return network