import torch
import numpy as np
from tools.utils import batch_input
from nn.distributions import CategoricalAction, MixtureAction, NormalAction
from nn.space import  Box, Discrete, MixtureSpace
from tools.config import Configurable
from .traj import Trajectory, DataBuffer
from tools.optim import LossOptimizer


def create_prior(space, device='cuda:0'):
    if isinstance(space, Box):
        zero = batch_input(np.zeros(space.shape), device=device)
        return NormalAction(zero, zero + 1.)
    elif isinstance(space, Discrete):
        assert isinstance(space, Discrete)
        return CategoricalAction(logits=torch.zeros((space.n,), device=device))
    elif isinstance(space, MixtureSpace):
        discrete = create_prior(space.discrete, device=device)
        continuous = create_prior(Box(-1, 1, (space.discrete.n, space.continuous.shape[0])), device=device)
        return MixtureAction(discrete, continuous)
    else:
        raise NotImplementedError



class Relbo(Configurable):
    def __init__(self,
            info_net,
            hidden_space,
            action_space,
            cfg=None,
            ent_z=1.,
            ent_a=1.,
            mutual_info=1.,
            reward=1.,
            prior=1.,
            learning_epoch=5,
     ) -> None:
        super().__init__()

        self.info_net = info_net
        self.info_net_optim = LossOptimizer(self.info_net, lr=1e-3)
        self.prior_z = create_prior(hidden_space)
        self.prior_a = create_prior(action_space)

    def compute_prior(self, a):
        return self.pi_a.log_prob(a)

    @torch.no_grad()
    def __call__(self, traj: Trajectory, batch_size):
        device = 'cuda:0'

        r = traj.get_tensor('r', device).sum(axis=-1) # TODO: try not to sum up all rewards ..
        ent_z = -traj.get_tensor('log_p_z', device)
        ent_a = -traj.get_tensor('log_p_a', device)
        prior = self.prior_a.log_prob(traj.get_tensor('a', device))
        mutual_info = traj.predict_value(['obs', 'a', 'z', 'timestep'], self.info_net, batch_size = batch_size)
        #TODO: test by varying batch size

        elbo = r * self._cfg.reward + mutual_info * self._cfg.mutual_info + \
            ent_z * self._cfg.ent_z + ent_a * self._cfg.ent_a + prior * self._cfg.prior

        return elbo[..., None]


    def learn(self, data: DataBuffer):
        # raise NotImplementedError
        for i in range(self._cfg.learning_epoch):
            for batch in data.loop_over('obs', 'a', 'z'):
                loss = -self.info_net(batch['obs'], batch['a'], batch['z'])
                loss = loss.mean()
                self.info_net_optim.optimize(loss)