from .rpgm import *
from torch import nn
from nn.distributions import DistHead

class Concatenate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(x, dim=-1), x[0]

class Extract(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[..., :self.index]


class SAC(Trainer):
    # SAC with single Q?
    def __init__(
            self, env: Union[GymVecEnv, TorchEnv], cfg=None, 

            head = DistHead.to_build(TYPE='Normal', linear=False, std_mode='fix_no_grad', std_scale=0.2), 
            weights=dict(state=0., prefix=1., value=1.), # predict reward + value from the o directly ..
            horizon=1,
            predict_Q=True,
        ):
        super().__init__(env)

    def make_network(self, obs_space, action_space):
        # let o_1 and s 1 be the concatenate of (s, a1)
        # predict the Q as the sum of r(s, a) + V(s, a)
        hidden_dim = 256
        latent_dim = 100 # encoding of state ..

        state_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        enc_s = Identity() #mlp(obs_space.shape[0], hidden_dim, latent_dim) # TODO: layer norm?
        enc_z = Identity()
        enc_a = Identity() #mlp(action_space.shape[0], hidden_dim, hidden_dim)

        init_h = Identity() #mlp(latent_dim, hidden_dim, hidden_dim) # reconstruct obs ..
        dynamics = Concatenate()

        #layer norm, if necesssary
        value = CatNet(
            Seq(mlp(state_dim + action_dim, hidden_dim, 1)),
            Seq(mlp(state_dim + action_dim, hidden_dim, 1)),
        )
        value_prefix = Seq(mlp(state_dim + action_dim, hidden_dim, 1))
        state_dec = Extract(state_dim) # should not be used ..

        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(state_dim, hidden_dim, head.get_input_dim()), head)
        network = GeneralizedQ(enc_s, enc_a, enc_z, pi_a, None, init_h, dynamics, state_dec,  value_prefix, value, cfg=self._cfg.qnet)
        network.apply(orthogonal_init)
        return network