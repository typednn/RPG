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


from tools.utils.scheduler import Scheduler


class EntropyLearner(Configurable):
    def __init__(
        self,
        space, name,
        cfg=None,
        coef=1.,
        target_mode='auto',
        target=None,
        lr=3e-4,
        device='cuda:0',
        schedule=Scheduler.to_build(
            TYPE='constant'
        )
    ):
        super().__init__(cfg)
        self.name = name

        if target is not None or target_mode == 'auto':
            self.log_alpha = torch.nn.Parameter(
                torch.zeros(1, requires_grad=(target is not None), device=device))
            if target == None:
                if isinstance(space, Box):
                    target = -space.shape[0]
                elif isinstance(space, Discrete):
                    target = space.n
                else:
                    raise NotImplementedError
            self.target  = target
            self.optim = LossOptimizer(self.log_alpha, lr=lr) #TODO: change the optim ..
        else:
            self.target = None
            self.schedule: Scheduler = Scheduler.build(cfg=schedule)

        
    def update(self, entropy):
        if self.target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self.target - entropy.detach()))
            self.optim.optimize(entropy_loss)
        else:
            self.schedule.step()

        logger.logkv_mean(self.name + '_alpha', float(self.alpha))
        logger.logkv_mean(self.name + '_ent', float(entropy.mean()))

    @property
    def alpha(self):
        if self.target is not None:
            return self.schedule.get() * self._cfg.coef
        return float(self.log_alpha.exp() * self._cfg.coef)
        

class PolicyLearner(LossOptimizer):
    def __init__(self, name, policy, cfg=None, ent=EntropyLearner.dc, freq=1):
        super().__init__(policy, cfg)
        self.name = name
        self.policy = policy
        self.ent = EntropyLearner(cfg=ent)
        self.freq = 1

    def update(self, rollout):
        # update policy and the entropy module ..
        loss, ent = self.policy.loss(rollout)
        logger.logkv_mean(self.name + '_loss', float(loss))
        self.optimize(loss)
        self.ent.update(ent)

    # @classmethod
    # def build_linear_agent_from_cfg(
    #     name, policy_type,
    #     state_dim, z_dim, hidden_dim, action_space,
    #     cfg=None, 
    # ):
    #     # make up for now ..
    #     from .soft_actor_critic import DiffPolicy
    #     policy = eval(policy_type)(state_dim, z_dim, hidden_dim, action_space)
    #     return PolicyLearner(name, policy, cfg)
    @classmethod
    def build_from_cfg(self, name, policy_cfg, learner_cfg, *args, **kwargs):
        from .soft_actor_critic import PolicyBase
        policy = PolicyBase.build(*args, cfg=policy_cfg, **kwargs)
        return PolicyLearner(name, policy, cfg=learner_cfg)