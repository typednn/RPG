import torch as th
from tools.config import Configurable
from tools.utils import batch_input
from nn.distributions import ActionDistr
from tools.optim import OptimModule as Optim
from .utils import minibatch_gen
from .traj import DataBuffer
from tools.utils import RunningMeanStd
from tools import dist_utils


class PolicyOptim(Optim):
    # no need to store distribution, we only need to store actions ..
    def __init__(
        self,
        actor,
        cfg=None,
        lr=5e-4,
        clip_param=0.2,
        entropy_coef=0.0,
        max_kl=0.1,
        #max_grad_norm=0.5,
        max_grad_norm=0.5,
        mode='step',

        entropy_target=None,
    ):
        super(PolicyOptim, self).__init__(actor)

        self.actor = actor
        self.clip_param = clip_param

        self.entropy_coef = entropy_coef
        self.entropy_target = entropy_target

        if self.entropy_coef > 0.:
            self.log_alpha = th.nn.Parameter(th.zeros(1, requires_grad=(self.entropy_target is not None)))

            if self.entropy_target is not None:
                self.entropy_target = entropy_target
                dist_utils.sync_networks(self.log_alpha)
                self.alpha_optim = th.optim.Adam([self.log_alpha], lr=lr, eps=self._cfg.eps)

    def optim_step(self):
        super().optim_step()

        if self.entropy_target is not None:

            dist_utils.sync_grads(self.log_alpha)
            if self._cfg.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.params, self._cfg.max_grad_norm)
            self.alpha_optim.step()

    def step(self, *args, backward=True, **data):
        assert self._cfg.mode == 'step', "please set the mode to be step to use this mode.. as we do not support accumulate_grad in this mode"
        # one step optimization
        if backward:
            self.optimizer.zero_grad()
            if self.entropy_target is not None:
                self.alpha_optim.zero_grad()
        loss, info = self._compute_loss(*args, backward=backward, **data)
        if backward:
            self.optim_step()
        return loss, info

    def get_entropy_coef(self):
        if self.entropy_coef > 0.:
            return self.log_alpha.exp().item() * self.entropy_coef
        return 0.

    def _compute_loss(self, obs, hidden, timestep, action, logp, adv, backward=True):
        device = self.actor.device
        action = batch_input(action, device)
        if hidden is not None:
            hidden = batch_input(hidden, device, dtype=hidden[0].dtype)
        timestep = batch_input(timestep, device)

        pd: ActionDistr = self.actor(obs, hidden, timestep)

        newlogp = pd.log_prob(action)
        device = newlogp.device

        adv = batch_input(adv, device).sum(axis=-1)
        logp = batch_input(logp, device)
        assert adv.shape == logp.shape

        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = th.exp(logratio)
        assert newlogp.shape == logp.shape
        assert adv.shape == ratio.shape, f"Adv shape is {adv.shape}, and ratio shape is {ratio.shape}"

        if self.clip_param > 0:
            pg_losses = -adv * ratio

            clipped_ratio = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            pg_losses2 = -adv * clipped_ratio
            pg_losses = th.max(pg_losses, pg_losses2)

            how_many_clipped = (clipped_ratio != ratio).float().mean()
                
        else:
            raise NotImplementedError

        # ---------------------------------------------------------------
        if self.entropy_coef > 0.:
            entropy = -newlogp.mean() * (self.log_alpha.exp()).detach()
            negent  = -entropy * self.entropy_coef
        else:
            negent = th.zeros(0., device=device)
        # -----------------------------------------------------------

        pg_losses = pg_losses.mean()

        loss = negent + pg_losses

        approx_kl_div = (ratio - 1 - logratio).mean().item()

        early_stop = self._cfg.max_kl is not None and (
            approx_kl_div > self._cfg.max_kl * 1.5)

        if self.entropy_target > 0.:
            # decrease, when entropy > self.entropy_target, positive
            # newlogp = - entropy_term
            alpha_loss = - (self.log_alpha.exp() * (self.entropy_target + newlogp).detach()).mean()
            if not early_stop and backward:
                alpha_loss.backward()

        if not early_stop and backward:
            loss.backward()
        else:
            pass

        output = {
            'entropy': entropy.item(),
            'negent': negent.item(),
            'pg': pg_losses.item(),
            'approx_kl': approx_kl_div,
            'loss': loss.item(),
            'clip_frac': how_many_clipped.item(),
        }
        if self._cfg.max_kl:
            from tools.dist_utils import get_world_size
            assert get_world_size() == 1
            output['early_stop'] = early_stop

        return loss, output


class CriticOptim(Optim):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5, mode='step'):
        super(CriticOptim, self).__init__(critic)
        self.critic = critic
        self.vfcoef = vfcoef

    def compute_loss(self, obs, hidden, timestep, vtarg):
        vpred = self.critic(obs, hidden, timestep)
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf


class PPOAgent(Configurable):
    def __init__(
        self, policy, critic,
        cfg=None,
        actor_optim=PolicyOptim.get_default_config(),
        critic_optim=None, 
        learning_epoch=10,

        rew_norm=True,
        adv_norm=True,
    ):
        super().__init__()

        self.policy = policy
        self.critic = critic

        self.actor_optim = PolicyOptim(self.policy, cfg=actor_optim)

        if critic_optim is None:
            critic_optim = dict(lr=actor_optim.lr)
        self.critic_optim = CriticOptim(self.critic, cfg=critic_optim)

        if rew_norm:
            self.rew_norm = RunningMeanStd(clip_max=10., last_dim=True)
        else:
            self.rew_norm = None

    def __call__(self, obs, hidden, timestep):
        return self.policy(obs, hidden, timestep=timestep)

    def value(self, obs, hidden, timestep):
        value =  self.critic(obs, hidden, timestep=timestep)
        if self.rew_norm is not None:
            value = value * self.rew_norm.std # denormalize it.
        return value

    def step(self, obs, hidden, timestep, action, log_p_a, adv, vtarg, logger_scope=None):
        actor_loss, actor_output = self.actor_optim.step(obs, hidden, timestep, action, log_p_a, adv)
        critic_loss, _ = self.critic_optim.step(obs, hidden, timestep, vtarg)

        if logger_scope is not None:
            from tools.utils import logger
            output = {logger_scope + k: v for k, v in actor_output.items()}
            output[logger_scope + 'critic_loss'] = critic_loss.item()
            output[logger_scope + 'actor_loss'] = actor_loss.item()
            logger.logkvs_mean(output)

        return 'early_stop' in actor_output and actor_output['early_stop']

    def learn(self, data: DataBuffer, batch_size, keys, logger_scope=None):
        stop = False

        if self.rew_norm is not None:
            vtarg = data['vtarg']
            self.rew_norm.update(vtarg)
            data['vtarg'] = vtarg / self.rew_norm.std
            data['adv'] = data['adv'] / self.rew_norm.std
            assert self.rew_norm.std.shape[-1] == vtarg.shape[-1]

        if self._cfg.adv_norm:
            adv = data['adv'].sum(axis=-1, keepdims=True)
            data['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        if logger_scope is not None:
            if len(logger_scope)>0:
                logger_scope = logger_scope + '/'

        for i in range(self._cfg.learning_epoch):
            n_batches = 0
            for batch in data.loop_over(batch_size):
                n_batches += 1
                if not stop:
                    stop = self.step(*[batch[i] for i in keys], logger_scope=logger_scope)
                    if stop:
                        break
            if stop:
                break