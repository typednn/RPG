import torch
from tools.utils import logger
from tools.nn_base import Network
from nn.distributions import DistHead, NormalAction
from tools.utils import Seq, mlp
from tools.optim import LossOptimizer
from tools.config import Configurable
from tools.utils import totensor
from nn.space import Discrete
from gym.spaces import Box


class InfoNet(Network):
    def __init__(self, 
                state_dim, action_space, hidden_dim, hidden_space, cfg=None, learn_posterior=False):
        super().__init__()
        action_dim = action_space.shape[0]
        from .hidden import HiddenSpace
        self.hidden: HiddenSpace = hidden_space
        self.info_net = Seq(mlp(state_dim + action_dim, hidden_dim, self.hidden.get_input_dim()))

        self.config = self.hidden._cfg

        if learn_posterior:
            self.posterior_z = Seq(mlp(state_dim, hidden_dim, self.hidden.get_input_dim()))

    def compute_feature(self, states, a_seq):
        states = states * self.config.obs_weight
        a_seq = (a_seq + torch.randn_like(a_seq) * self.config.noise)
        a_seq = a_seq * self.config.action_weight
        return self.info_net(states, a_seq)

    def get_state_seq(self, traj):
        if self.config.use_next_state:
            return traj['state'][1:]
        return traj['state'][:-1]

    def forward(self, traj, mode='likelihood'):
        states = self.get_state_seq(traj)
        a_seq = traj['a']
        z_seq = traj['z']

        if mode != 'reward':
            states = states.detach()
            a_seq = a_seq.detach()
            z_seq = z_seq.detach()

        inp = self.compute_feature(states, a_seq)
        if mode != 'sample':
            t = traj['timestep']
            if mode == 'likelihood':
                return self.hidden.likelihood(inp, z_seq, timestep=t)[..., None]
            elif mode == 'reward':
                return self.hidden.reward(inp, z_seq, timestep=t)[..., None]
        else:
            return self.hidden.sample(inp)

    def enc_s(self, obs, timestep):
        return self.enc_s(obs, timestep=timestep)

    def get_posterior(self, state, z=None):
        #return self.posterior_z(states)
        inp = self.posterior_z(state)
        if z is not None:
            return self.hidden.likelihood(inp, z, timestep=None)
        else:
            return self.hidden.sample(inp)


from tools.utils.scheduler import Scheduler
class InfoLearner(LossOptimizer):
    def __init__(self, state_dim, action_space, hidden_space, cfg=None,
                 net=InfoNet.dc,
                 coef=1.,
                 weight=Scheduler.to_build(TYPE='constant'),
                 hidden_dim=256, learn_posterior=False
        ):
        net = InfoNet(
            state_dim, action_space, hidden_dim, hidden_space, cfg=net,
            learn_posterior=learn_posterior
        ).cuda()
        self.coef = coef
        self.info_decay: Scheduler = Scheduler.build(weight)
        self.learn_posterior = learn_posterior
        super().__init__(net)
        self.net = net

    @classmethod
    def build_from_cfgs(self, net_cfg, learner_cfg, *args, **kwargs):
        net = InfoNet(*args, cfg=net_cfg, **kwargs)
        return InfoLearner(net, cfg=learner_cfg)

    def get_coef(self):
        return self.coef * self.info_decay.get()

    def intrinsic_reward(self, traj):
        info_reward = self.net(traj, mode='reward')
        return 'info', info_reward * self.get_coef()
    
    def update(self, rollout):
        z_detach = rollout['z'].detach()

        mutual_info = self.net(rollout, mode='likelihood').mean()
        if self.learn_posterior:
            posterior = self.net.get_posterior(rollout['state'][1:].detach()).log_prob(z_detach).mean()

        self.optimize(-mutual_info - posterior)

        # TODO: estimate the posterior
        logger.logkv_mean('info_ce_loss', float(-mutual_info))
        logger.logkv_mean('info_posterior_loss', float(-posterior))

        self.info_decay.step()
        logger.logkv_mean('info_decay', self.info_decay.get())

    def sample_z(self, states, a):
        #print(states.shape)
        #return self.net.get_posterior(states).sample()
        return self.net(
            {
                'state': states,
                'a': a[None,:],
                'z': a[None, :] * 0 # fake z
            },
            mode='sample'
        )[0]

    def sample_z_by_posteior(self, states):
        return self.net.get_posterior(states)[0]