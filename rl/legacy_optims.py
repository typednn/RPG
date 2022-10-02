class BC(Optim):
    def __init__(self, actor, cfg=None, lr=5e-4):
        super(BC, self).__init__(actor)
        self.actor = actor
        self.update_steps = 0

    def step(self, obs, action, phase='train'):
        raise NotImplementedError
        assert phase in [
            'train', 'valid'], f'unrecognized phase {phase}, backward + optimize only if in training phase'
        if phase == 'train':
            self.actor.train()
        else:
            self.actor.eval()

        with torch.set_grad_enabled(phase == 'train'):
            pd: ActionDistr = self.actor(obs)
            logp = - pd.log_prob(action)
            loss = logp.mean()
            if phase == 'train':
                self.optimize(loss)
                self.update_steps += 1

        return {
            f'logp_{phase}': loss.item()
        }



class GDOptim(Optim):
    # https://github.com/hzaskywalker/PlasticineLabV2/blob/20f190700a941115fbc9e47af9e76d7c571a0481/plb/algorithms/ppg/diff_roller.py
    # large gradient step here ..
    def __init__(self, actor, cfg=None, lr=3e-4) -> None:
        super().__init__(actor)
        self.actor = actor
        raise NotImplementedError

    def step(self, obs, action, grad):
        from .models import NormalAction

        pd: NormalAction = self.actor(obs)
        action = pd.batch_action(action)
        # grad has exactly the same shape as action
        grad = -pd.batch_action(grad) # 
        print(grad.mean(axis=0))

        assert action.shape == pd.mean.shape == pd.scale.shape
        theta = ((action - pd.mean) / pd.scale).detach()

        # grad is the grad of the neg-reward

        # action = theta * scale + mu
        # we should
        loss = (pd.mean * grad.detach() + pd.scale *
                (grad/theta).detach()).sum(axis=-1).mean()

        self.optimize(loss)
        logger.logkvs_mean(
            {'gd_optim': loss.item()}
        )