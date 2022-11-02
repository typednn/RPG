# multimodal actor
import numpy as np
from .oc import OptionCritic
from .oc import *
from tools.utils import Identity
from nn.distributions import DeterminisiticAction


class IdentityActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, z):
        return DeterminisiticAction(z)

class DiscreteActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, z):
        idx = z.argmax(dim=-1)
        idx = idx.float() / z.shape[-1] * np.pi * 2
        a = torch.stack((torch.cos(idx), torch.sin(idx)), dim=-1)
        return DeterminisiticAction(a)

class TanhActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return DeterminisiticAction(torch.tanh(z))



class MultiModal(OptionCritic):
    def __init__(
        self, env,
        cfg=None,
        z_dim=0,
        z_cont_dim=10,
        z_head=None,
        option_mode='everystep',
        actor_mode ='identify',
    ):
        """
        Actor: 
            ignore z: z is used and a is independent with z.
            identify: z is in fact the true action, and $a=z$.
            deterministic: a is a deterministic function of z, equiping with a cycle consistency.
            double randomness: a is also a stochastic policy
        Learning:
            pg: policy gradient
            gd: gradient descent
        """
        if actor_mode == 'identify':
            z_cont_dim = env.action_space.shape[0]
        super().__init__(env, z_cont_dim=z_cont_dim)

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        latent_dim = 100
        assert self._cfg.option_mode == 'everystep'

        action_dim = action_space.shape[0]

        # TODO: layer norm?
        from .utils import ZTransform
        enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim))
        # latent_dim = obs_space.shape[0]
        # enc_s = TimedSeq(Identity())
        enc_z = ZTransform(z_space)
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)

        value_prefix = Seq(mlp(latent_dim + hidden_dim, hidden_dim, 1))
        done_fn = mlp(hidden_dim, hidden_dim, 1) if self._cfg.have_done else None
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..


        v_in = latent_dim
        value = CatNet(
            Seq(mlp(v_in, hidden_dim, 1)),
            Seq(mlp(v_in, hidden_dim, 1)),
        ) #layer norm, if necesssary

        from rpg.utils import config_hidden
        zhead_cfg = config_hidden(self._cfg.z_head, z_space)
        if self._cfg.actor_mode == 'identify':
            pi_a = IdentityActor()
            zhead = DistHead.build(action_space, cfg=self._cfg.head)
            import numpy as np
            zhead.LOG_STD_MAX = np.log(1.)
            zhead.LOG_STD_MIN = np.log(0.01)
        elif self._cfg.actor_mode == 'discrete':
            zhead = DistHead.build(z_space, cfg=zhead_cfg)
            pi_a = DiscreteActor()
        else:
            zhead = DistHead.build(z_space, cfg=zhead_cfg)

            z_dim = z_space.inp_shape[0]
            latent_z_dim = z_dim

            if self._cfg.actor_mode == 'determinstic':
                pi_a = Seq(
                    mlp(v_in + latent_z_dim, hidden_dim, action_space.shape[0]),
                    TanhActor(),
                )
            else:
                head = DistHead.build(action_space, cfg=self._cfg.head)
                pi_a = Seq(mlp(latent_dim + latent_z_dim, hidden_dim, head.get_input_dim()), head)


        #zhead = DistHead.build(z_space, cfg=config_hidden(self._cfg.z_head, z_space))
        backbone = Seq(mlp(v_in, hidden_dim, hidden_dim))
        pi_z = OptionNet(zhead, backbone, hidden_dim, done_mode=self._cfg.option_mode, detach=not self._cfg.z_grad)


        self.info_net = IntrinsicReward(latent_dim, action_dim, hidden_dim, self.z_space, entropy_coef=self._cfg.entz_coef, cfg=self._cfg.ir)

        network = GeneralizedQ(
            enc_s, enc_a, enc_z, pi_a, pi_z, init_h, dynamics, state_dec, value_prefix, value, done_fn,
            self.info_net,

            cfg=self._cfg.qnet,
            horizon=self._cfg.horizon,
            markovian=True,
        )
        network.apply(orthogonal_init)
        return network