import torch as th
from tools.config import Configurable
from tools.utils import batch_input
from nn.distributions import ActionDistr
from tools.utils import logger
from tools.optim import OptimModule as Optim
from .utils import compute_gae_by_hand
from .traj import DataBuffer
from tools.utils import RunningMeanStd
from tools import dist_utils
from tools.constrained_optim import COptim

# class CPO(COptim):
#     # constrained optimizer ..
#     def __init__(self, network, cf=None, clip_param=0.2) -> None:
#         super().__init__(network, 1)


class PolicyOptim(Optim):
    # no need to store distribution, we only need to store actions ..
    def __init__(
        self,
        actor,
        cfg=None,
        lr=3e-4,
        clip_param=0.2,
        max_kl=0.1,
        #max_grad_norm=0.5,
        max_grad_norm=0.5,
        mode='step',

    ):
        super(PolicyOptim, self).__init__(actor)

        self.actor = actor
        self.clip_param = clip_param



    def _compute_loss(self, obs, hidden, timestep, action, logp, adv, entropy_coef, backward=True):
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


        approx_kl_div = (ratio - 1 - logratio).mean().item()

        early_stop = self._cfg.max_kl is not None and (
            approx_kl_div > self._cfg.max_kl * 1.5)

        # ---------------------------------------------------------------
        if entropy_coef > 0.:
            # directly optimize the policy entropy ..
            entropy = pd.entropy().mean()
            negent  = -entropy * entropy_coef
        else:
            entropy = 0.
            negent = th.tensor(0., device=device)
        # -----------------------------------------------------------

        pg_losses = pg_losses.mean()
        loss = negent + pg_losses

        if not early_stop and backward:
            loss.backward()
        else:
            pass

        output = {
            'entropy': float(entropy),
            'negent': negent.item(),
            'pg': pg_losses.item(),
            'approx_kl': approx_kl_div,
            'loss': loss.item(),
            'clip_frac': how_many_clipped.item(),
            'entropy_coef_after_norm': float(entropy_coef),
        }
        if self._cfg.max_kl:
            from tools.dist_utils import get_world_size
            assert get_world_size() == 1
            output['early_stop'] = early_stop

        return loss, output


class CriticOptim(Optim):
    def __init__(self, critic, cfg=None,
                 lr=3e-4, vfcoef=0.5, mode='step'):
        super(CriticOptim, self).__init__(critic)
        self.critic = critic
        self.vfcoef = vfcoef

    def compute_loss(self, obs, hidden, timestep, vtarg):
        vpred = self.critic(obs, hidden, timestep)
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf


class EntropyOptim(Optim):
    def __init__(self, cfg=None, coef=0.0, target=None, lr=0.001, mode='step'):
        super().__init__(th.nn.Parameter(th.zeros(1, requires_grad=(target is not None), device='cuda:0')))
        self.log_alpha = self.network

    def __call__(self):
        return self._cfg.coef * self.log_alpha.exp().detach()

    def compute_loss(self, entropy):
        # if self._cfg.target > entropy, increase log alpha
        #print(self._cfg.target - entropy)
        logger.logkv_mean('gap', float(self._cfg.target - entropy))
        return - (self.log_alpha.exp() * (self._cfg.target - entropy))

    def step(self, entropy):
        if self._cfg.target is not None:
            return super().step(entropy)
        return None
        

class PPOAgent(Configurable):
    def __init__(
        self, policy, critic,
        cfg=None,
        actor_optim=PolicyOptim.dc,
        critic_optim=None, 
        entropy=EntropyOptim.dc,
        learning_epoch=10,

        rew_norm=True,
        adv_norm=True,

        # gae config
        gamma=0.995, lmbda=0.97,
        ignore_done=False

    ):
        super().__init__()

        self.policy = policy
        self.critic = critic

        self.actor_optim = PolicyOptim(self.policy, cfg=actor_optim)

        if critic_optim is None:
            critic_optim = dict(lr=actor_optim.lr)
        self.critic_optim = CriticOptim(self.critic, cfg=critic_optim)

        self.ent_optim = EntropyOptim(cfg=entropy)


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

    def entropy(self, obs, hidden, timestep):
        return self.policy(obs, hidden, timestep=timestep).entropy()[..., None]

    def ent_coef(self):
        return self.ent_optim()

    def gae(
        self, vpred, next_vpred, reward, done, truncated, entropy, next_entropy,
    ):
        if self._cfg.ignore_done:
            done = done * 0

        ent_coef = self.ent_coef()
        if ent_coef > 0.:
            logger.logkv_mean('ent coef', float(ent_coef))

            vpred[..., -1:] += ent_coef * entropy
            next_vpred[..., -1:] += ent_coef * next_entropy

            assert entropy.shape[:-1] == reward.shape[:-1]
            reward = th.cat((reward, ent_coef * entropy), dim=-1)

        assert vpred.shape == next_vpred.shape == reward.shape, "vpred and next_vpred must be the same "\
            "length as reward, but got {} and {} and {}".format(vpred.shape, next_vpred.shape, reward.shape)

        adv = compute_gae_by_hand(
            reward, vpred, next_vpred, done, truncated,
            gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, mode='exact'
        )

        if ent_coef > 0.:
            vpred[..., -1:] -= ent_coef * entropy

        vtarg = vpred + adv

        return dict(adv=adv, vtarg = vtarg)


    def normalize(self, data, logger_scope):
        if self.rew_norm is not None:
            vtarg = data['vtarg']
            self.rew_norm.update(vtarg)

            assert self.rew_norm.std.shape[-1] == vtarg.shape[-1]
            data['vtarg'] = vtarg / self.rew_norm.std
            # data['adv'] = data['adv'] / self.rew_norm.std # 
            # print(self.rew_norm.std)

            for idx, i in enumerate(self.rew_norm.std.reshape(-1)):
                logger.logkv_mean(f'{logger_scope}rew{idx}_std', float(i))

        if self._cfg.adv_norm:
            adv = data['adv'].sum(axis=-1, keepdims=True)
            self.adv_std = float(adv.std()) + 1e-8
            data['adv'] = (adv - adv.mean()) / self.adv_std
            logger.logkv_mean(logger_scope + 'adv_std', self.adv_std)


    def learn_step(self, obs, hidden, timestep, action, log_p_a, adv, vtarg, logger_scope=None):
        actor_loss, actor_output = self.actor_optim.step(
            obs, hidden, timestep, action, log_p_a, adv, entropy_coef=self.ent_coef() / self.adv_std)
        critic_loss, _ = self.critic_optim.step(
            obs, hidden, timestep, vtarg)
        self.ent_optim.step(actor_output['entropy'])


        if logger_scope is not None:
            output = {logger_scope + k: v for k, v in actor_output.items()}
            output[logger_scope + 'critic_loss'] = critic_loss.item()
            output[logger_scope + 'actor_loss'] = actor_loss.item()
            logger.logkvs_mean(output)

        return 'early_stop' in actor_output and actor_output['early_stop']


    def learn(self, data: DataBuffer, batch_size, keys, logger_scope=None):

        stop = False
        if logger_scope is not None:
            if len(logger_scope)>0:
                logger_scope = logger_scope + '/'
        self.normalize(data, logger_scope)

        for i in range(self._cfg.learning_epoch):
            n_batches = 0
            for batch in data.loop_over(batch_size):
                n_batches += 1
                if not stop:
                    stop = self.learn_step(*[batch[i] for i in keys], logger_scope=logger_scope)
                    if stop:
                        break
            if stop:
                break