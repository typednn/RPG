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
                state_dim, action_space, hidden_dim, hidden_space,
                cfg=None,
                mutual_info_weight=0., backbone=None, 
                action_weight=1., noise=0.0, obs_weight=1.,
                head=None, use_next_state=False, #, epsilon=0.2,
                # std_mode='fix_no_grad',
                ):
        super().__init__()
        action_dim = action_space.shape[0]
        from .hidden import HiddenSpace
        self.hidden: HiddenSpace = hidden_space
        self.info_net = Seq(mlp(state_dim + action_dim, hidden_dim, self.hidden.get_input_dim()))

    def compute_feature(self, states, a_seq):
        states = states * self._cfg.obs_weight
        a_seq = (a_seq + torch.randn_like(a_seq) * self._cfg.noise)
        a_seq = a_seq * self._cfg.action_weight
        return self.info_net(states, a_seq)

    def get_state_seq(self, traj):
        if self._cfg.use_next_state:
            return traj['state'][1:]
        return traj['state'][:-1]

    def forward(self, traj, detach=False):
        states = self.get_state_seq(traj)
        a_seq = traj['a']
        z_seq = traj['z']

        if detach:
            states = states.detach()
            a_seq = a_seq.detach()
            z_seq = z_seq.detach()

        inp = self.compute_feature(states, a_seq)
        t = traj['init_timestep']
        ts = []
        for _ in range(len(inp)):
            ts.append(t)
            t = t + 1
        t = torch.stack(ts)
        if detach:
            info = self.hidden.likelihood(inp, z_seq, timestep=t)
        else:
            info = self.hidden.reward(inp, z_seq, timestep=t)
        return info[..., None]

    def enc_s(self, obs, timestep):
        return self.enc_s(obs, timestep=timestep)

    # def get_posterior(self, states):
    #     return self.posterior_z(states)


from tools.utils.scheduler import Scheduler
class InfoLearner(LossOptimizer):
    def __init__(self, state_dim, action_space, hidden_space, cfg=None,
                 net=InfoNet.dc,
                 coef=1.,
                 weight=Scheduler.to_build(TYPE='constant'),
                 hidden_dim=256,
        ):
        net = InfoNet(
            state_dim, action_space, hidden_dim, hidden_space, cfg=net
        ).cuda()
        self.coef = coef
        self.info_decay: Scheduler = Scheduler.build(weight)
        super().__init__(net)
        self.net = net

    @classmethod
    def build_from_cfgs(self, net_cfg, learner_cfg, *args, **kwargs):
        net = InfoNet(*args, cfg=net_cfg, **kwargs)
        return InfoLearner(net, cfg=learner_cfg)

    def get_coef(self):
        return self.coef * self.info_decay.get()

    def intrinsic_reward(self, traj):
        info_reward = self.net(traj, detach=False)
        return 'info', info_reward * self.get_coef()
    
    def update(self, rollout):
        z_detach = rollout['z'].detach()

        mutual_info = self.net(rollout, detach=True).mean()
        #posterior = self.net.get_posterior(rollout['state'][1:].detach()).log_prob(z_detach).mean()
        posterior = 0. # TODO: adding this later. if necessary

        self.optimize(-mutual_info - posterior)

        logger.logkv_mean('info_ce_loss', float(-mutual_info))
        logger.logkv_mean('info_posterior_loss', float(-posterior))

        self.info_decay.step()
        logger.logkv_mean('info_decay', self.info_decay.get())

    def sample_z(self, states):
        #print(states.shape)
        return self.net.get_posterior(states).sample()