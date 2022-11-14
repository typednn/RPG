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


class EntropyLearner(Configurable):
    def __init__(self, space, cfg=None, coef=1., target_mode='auto', target=None, lr=3e-4, device='cuda:0'):
        super().__init__(cfg)

        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=(target is not None), device=device))
        if target is not None or target_mode == 'auto':
            if target == None:
                self._cfg.defrost()
                if isinstance(space, Box):
                    target = -space.shape[0]
                else:
                    #raise NotImplementedError
                    target = space.n
                self._cfg.target = target
            self.optim = LossOptimizer(self.log_alpha, lr=lr) #TODO: change the optim ..
        
    def update(self, entropy):
        if self._cfg.target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self._cfg.target - entropy.detach()))
            self.optim.optimize(entropy_loss)

    @property
    def alpha(self):
        return float(self.log_alpha.exp() * self._cfg.coef)

            
class InfoNet(Network):
    def __init__(self, 
                state_dim, action_dim, hidden_dim, hidden_space,
                cfg=None,
                mutual_info_weight=0., backbone=None, 
                action_weight=1., noise=0.0, obs_weight=1., head=None, use_next_state=False, epsilon=0.,
                ):
        super().__init__(cfg)

        zhead = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        backbone = Seq(mlp(state_dim + action_dim, hidden_dim, zhead.get_input_dim()))
        self.zhead = zhead
        self.info_net = Seq(backbone, zhead)

        zhead2 = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        backbone = Seq(mlp(state_dim, hidden_dim, zhead2.get_input_dim()))
        self.posterior_z = Seq(backbone, zhead) # the posterior of p(z|s), for off-policy training..


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


    def enc_s(self, obs, timestep):
        return self.enc_s(obs, timestep=timestep)

    def get_posterior(self, states):
        return self.posterior_z(states)


from tools.utils.scheduler import Scheduler

class IntrinsicReward(Configurable):
    def __init__(
        self, enta, entz, info_net: InfoNet, optim_cfg,
        cfg=None,
        entz_decay=Scheduler.to_build(TYPE='constant'),
        info_decay=Scheduler.to_build(TYPE='constant'),
    ):
        super().__init__()
        self.enta = enta
        self.entz = entz
        self.info_net = info_net

        self.entz_decay = Scheduler.build(entz_decay)
        self.info_decay = Scheduler.build(info_decay)
        if info_net is not None:
            # import copy
            # self.info_target = copy.deepcopy(info_net)
            self.info_target = info_net

            self.info_optim = LossOptimizer(
                info_net,
                cfg = optim_cfg
            )


    def get_ent_from_traj(self, traj):
        return -traj['logp_a'], traj['ent_z']

    def ema(self, tau):
        #from tools.utils import ema
        # if self.info_net is not None:
        #     ema(self.info_net, self.info_target, tau)
        pass

    def estimate_unscaled_rewards(self, traj):
        reward = traj['reward']

        enta, entz = self.get_ent_from_traj(traj)
        enta = enta * self.enta.alpha
        entz = entz * self.entz.alpha * self.entz_decay.get()

        if self.info_net is not None:
            info_reward = self.info_target(
                traj, detach=False) * self.info_target._cfg.mutual_info_weight * self.info_decay.get() # in case of discrete .. # only use the target to compute
        else:
            info_reward = enta * 0.
        entropies = torch.cat((enta, entz, info_reward), dim=-1)

        logger.logkv_mean('reward_enta', enta.mean().item())
        logger.logkv_mean('reward_entz', entz.mean().item())
        logger.logkv_mean('reward_info', info_reward.mean().item())
        logger.logkv_mean('decay_z', self.entz_decay.get())
        logger.logkv_mean('decay_info', self.info_decay.get())

        assert reward.shape == info_reward.shape
        reward = reward # + info_reward
        return reward, entropies, {}

    def update(self, traj):
        #s_seq = self.info_net.get_state_seq(samples).detach()
        self.entz_decay.step()
        self.info_decay.step()
        if self.info_net is not None:
            z_detach = traj['z'].detach()
            mutual_info = self.info_net(traj, detach=True).mean()
            posterior = self.info_net.get_posterior(traj['state'][1:].detach()).log_prob(z_detach).mean()
            self.info_optim.optimize(-mutual_info - posterior)

            logger.logkv_mean('info_ce_loss', float(-mutual_info))
            logger.logkv_mean('info_posterior_loss', float(-posterior))

    def sample_posterior_z(self, enc_s, obs_seq, action, timestep):
        # sample by posterior for the first obs  
        # using s, a and info net to sample for the remaining ..
        s = enc_s(obs_seq, timestep=timestep)
        assert action.shape[0] == obs_seq.shape[0] - 1
        assert len(s.shape) == 3
        if self.info_net is None:
            assert action.dtype == torch.long
            z0 = torch.zeros(s.shape[1:-1], dtype=torch.long, device=s.device)
            return torch.cat((z0[None, :], action))
        return self.info_net.get_posterior(s).sample()[0]
        # prev0 = self.info_net.get_posterior(s[:1]).sample()[0]
        # states = self.info_net.get_state_seq({'state': s})
        # prev1 = self.info_net.compute_info_dist(states, action).sample()[0]
        # return torch.cat((prev0, prev1))
        # sample latent z uniformly ..
        # x = torch.zeros(s.shape[:-1] +(self.info_net.zhead.get_input_dim(),), dtype=torch.float, device=s.device)
        # return self.info_net.zhead(x).sample()[0]