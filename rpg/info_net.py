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
        hidden_space: HiddenSpace
        self.zhead = hidden_space.make_info_head(cfg=head)
        backbone = Seq(mlp(state_dim + action_dim, hidden_dim, self.zhead.get_input_dim()))
        self.info_net = Seq(backbone, self.zhead)

        # # the posterior of p(z|s), for off-policy training..
        # zhead2 = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        # backbone = Seq(mlp(state_dim, hidden_dim, zhead2.get_input_dim()))
        # self.posterior_z = Seq(backbone, zhead) 

    def compute_info_dist(self, states, a_seq):
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
        info =  self.compute_info_dist(
            states, a_seq).log_prob(z_seq)
        return info[..., None]

    def config_head(self, hidden_space):
        from tools.config import merge_inputs
        discrete = dict(TYPE='Discrete', epsilon=self._cfg.epsilon)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode=self._cfg.std_mode, std_scale=0.3989)

        if isinstance(hidden_space, Discrete):
            head = discrete
        elif isinstance(hidden_space, Box):
            head = continuous
        else:
            raise NotImplementedError
        if self._cfg.head is not None:
            head = merge_inputs(head, **self._cfg.head)
        return head

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