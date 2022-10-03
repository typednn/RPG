import torch
from nn.distributions import ActionDistr, Normal, Discrete as Categorical
from solver.actor import Actor as BaseNet
from gym.spaces import Box, Discrete
from tools.nn_base import Network
from tools.config import as_builder


class Policy(BaseNet):
    def __init__(self, obs_space, z_space, action_space, cfg=None):
        self.z_space = z_space
        if z_space is not None:
            obs_space = (obs_space, z_space)

        if isinstance(action_space, Box):
            assert len(action_space.shape) == 1, "We do not support multi-dimensional actions yet. Please flatten them or serialize them first."
        else:
            assert cfg.head.TYPE == 'Discrete', f"{cfg.head.TYPE} not suitable for {action_space}"

        BaseNet.__init__(self, obs_space, action_space)

    def forward(self, state, hidden, timestep=None):
        if self.z_space is None:
            return super().forward(state, timestep=timestep)
        else:
            return super().forward((state, hidden), timestep=timestep)


class Critic(BaseNet):
    def __init__(self, obs_space, z_space, dim, cfg=None):
        self.z_space = z_space
        if z_space is not None:
            obs_space = (obs_space, z_space)
        BaseNet.__init__(self, obs_space, Box(-1, 1, (dim,)), head=dict(TYPE='Deterministic'))

    def forward(self, state, hidden, timestep=None):
        if self.z_space is None:
            return super().forward(state, timestep=timestep).rsample()[0]
        else:
            return super().forward((state, hidden), timestep=timestep).rsample()[0]


@as_builder
class InfoNet(Network):
    # should use a transformer instead ..
    def __init__(self, obs_space, action_space, hidden_space, cfg=None, backbone=None, action_weight=1., noise=0.0, obs_weight=1.):
        super().__init__()
        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=1.)

        if isinstance(hidden_space, Discrete):
            z_head = discrete
        elif isinstance(hidden_space, Box):
            z_head = continuous
        else:
            z_head = dict(
                TYPE="Mixture",
                discrete=discrete,
                continuous=continuous,
            )
        
        self.main = BaseNet((obs_space, action_space), hidden_space, backbone=backbone, head=z_head).cuda()
        self.preprocess = None

    def forward(self, s, a, z, timestep):
        #if timestep is None:
        #    timestep = torch.arange(len(s), device=self.device)
        from tools.utils import dmul, dshape

        if self.preprocess is not None:
            s = self.preprocess(s)
        s = self.batch_input(s)
        a = self.batch_input(a)
        z = self.batch_input(z)

        return self.main(
            (dmul(s, self._cfg.obs_weight),
             (a + torch.randn_like(a) * self._cfg.noise) * self._cfg.action_weight)
        ).log_prob(z)
