import torch as th
from tools.utils import batch_input
from nn.distributions import ActionDistr
from tools.optim import OptimModule as Optim


class PPO(Optim):
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
        super(PPO, self).__init__(actor)

        self.actor = actor
        self.entropy_coef = entropy_coef
        self.clip_param = clip_param

    def __call__(self, obs, **kwargs):
        return super().__call__(obs, **kwargs)

    def _compute_loss(self, obs, hidden, action, logp, adv, backward=True):
        pd: ActionDistr = self.actor((obs, hidden))

        newlogp = pd.log_prob(action)
        device = newlogp.device

        adv = batch_input(adv, device)
        logp = batch_input(logp, device)

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
            return early_stop
        return False
