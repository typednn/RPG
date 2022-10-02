import torch as th
from tools.config import Configurable
from tools.utils import batch_input
from nn.distributions import ActionDistr
from tools.optim import OptimModule as Optim
from .utils import minibatch_gen


class PolicyOptim(Optim):
    # no need to store distribution, we only need to store actions ..
    def __init__(self,
                 actor,
                 cfg=None,
                 lr=5e-4,
                 clip_param=0.2,
                 entropy_coef=0.0,
                 max_kl=None,
                 max_grad_norm=0.5,
                 mode='step',
                 ):
        super(PolicyOptim, self).__init__(actor)

        self.actor = actor
        self.entropy_coef = entropy_coef
        self.clip_param = clip_param

    def _compute_loss(self, obs, hidden, timestep, action, logp, adv, backward=True):
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
            pg_losses2 = -adv * \
                th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            pg_losses = th.max(pg_losses, pg_losses2)
        else:
            raise NotImplementedError

        entropy = pd.entropy(sum=True)
        assert entropy.shape == pg_losses.shape

        pg_losses, entropy = pg_losses.mean(), entropy.mean()
        negent = -entropy * self.entropy_coef

        loss = negent + pg_losses

        approx_kl_div = (ratio - 1 - logratio).mean().item()

        early_stop = self._cfg.max_kl is not None and (
            approx_kl_div > self._cfg.max_kl * 1.5)

        if not early_stop and backward:
            loss.backward()
        else:
            pass

        output = {
            'entropy': entropy.item(),
            'negent': negent.item(),
            'pi': loss.item(),
            'pg': pg_losses.item(),
            'approx_kl': approx_kl_div,
        }
        if self._cfg.max_kl:
            from tools.dist_utils import get_world_size
            assert get_world_size() == 1
            #TODO: need to sync across multiple processes
            return early_stop
        return False



class CriticOptim(Optim):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5, mode='step'):
        super(CriticOptim, self).__init__(critic)
        self.critic = critic
        #self.optim = make_optim(critic.parameters(), lr)
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
        critic_optim=None
    ):
        super().__init__()

        self.policy = policy
        self.critic = critic

        self.actor_optim = PolicyOptim(self.policy, cfg=actor_optim)

        if critic_optim is None:
            critic_optim = dict(lr=actor_optim.lr)
        self.critic_optim = CriticOptim(self.critic, cfg=critic_optim)

    def __call__(self, obs, hidden, timestep):
        return self.policy(obs, hidden, timestep=timestep)

    def value(self, obs, hidden, timestep):
        return self.critic(obs, hidden, timestep=timestep)

    def stpe(self, obs, hidden, timestep, action, log_p_a, adv, vtarg):
        stop = self.actor_optim.step(obs, hidden, timestep, action, log_p_a, adv)
        self.critic_optim.step(obs, hidden, timestep, vtarg)
        return stop

    def learn(self, data, batch_size, keys):
        stop = False
        for i in range(5):
            for batch in minibatch_gen(data, batch_size):
                if not stop:
                    stop = self.step(*[batch[i] for i in keys])
            if stop:
                break