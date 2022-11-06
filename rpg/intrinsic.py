import torch
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
                action_weight=1., noise=0.0, obs_weight=1., head=None, use_next_state=False,
                ):
        super().__init__(cfg)

        zhead = DistHead.build(hidden_space, cfg=self.config_head(hidden_space))
        backbone = Seq(mlp(state_dim + action_dim, hidden_dim, zhead.get_input_dim()))
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
            states, a_seq).log_prob(z_seq) * self._cfg.mutual_info_weight # in case of discrete ..
        return info[..., None]

    def config_head(self, hidden_space):
        from tools.config import merge_inputs
        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
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


class IntrinsicReward:
    def __init__(
        self, enta, entz, info_net: InfoNet, optim_cfg,
    ):
        self.enta = enta
        self.entz = entz
        self.info_net = info_net
        self.info_optim = LossOptimizer(
            info_net,
            cfg = optim_cfg
        )


    def get_ent_from_traj(self, traj):
        return -traj['logp_a'], traj['ent_z']

    def estimate_unscaled_rewards(self, traj):
        reward = traj['reward']

        enta, entz = self.get_ent_from_traj(traj)
        enta = enta * self.enta.alpha
        entz = entz * self.entz.alpha

        entropies = torch.cat((enta, entz), dim=-1)
        reward = reward + self.info_net(traj, detach=False)
        return reward, entropies, {}

    def update(self, traj):
        enta, entz = self.get_ent_from_traj(traj)
        self.enta.update(enta)
        self.entz.update(entz)

        #s_seq = self.info_net.get_state_seq(samples).detach()
        z_detach = traj['z'].detach()
        mutual_info = self.info_net(traj, detach=True).mean()
        posterior = self.info_net.get_posterior(traj['state'][1:].detach()).log_prob(z_detach).mean()
        self.info_optim.optimize(- mutual_info - posterior)

    def sample_posterior_z(self, enc_s, obs, timestep):
        s = enc_s(obs, timestep=timestep)
        return self.info_net.get_posterior(s).sample()[0]