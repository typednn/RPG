# mixture of gaussian libraries 

# p(a_o|s) = exp(R(a)) -> p(a_o|s) = log p(a_o|z) - p(z|s)

# \nabla p(z) R(f(z)) = E_z R(f(z))\nabla \log p(z) + \nabla R(f(z;theta))

#TODO: VAE, REINFORCE and mean policy  ..
#NOTE: we can't use either MOE or Gumbel Max, because gradient may not contain the information we want; 
#   however, empirically, gumbelmax may performs well due to SGD and suitable initialization ..

"""
loss = (pd.mean * grad.detach() + pd.scale *
        (grad/theta).detach()).sum(axis=-1).mean()
"""
import torch
from .normal import NormalAction
from torch.distributions import Categorical


class GMMAction(NormalAction):
    def __init__(self, log_mu, loc, scale, gumbel=False):
        super().__init__(loc, scale)
        assert log_mu.shape[:2] == loc.shape[:2] and len(log_mu.shape) == 2, f"{log_mu.shape}, {loc.shape}, {scale.shape}"
        self.gumbel = gumbel
        self.log_mu = log_mu
        self.mode = Categorical(logits=log_mu)
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size(), *args, **kwargs):
        action = self.dist.rsample(sample_shape, *args, **kwargs)
        if self.gumbel:
            if not isinstance(sample_shape, torch.Size):
                sample_shape = torch.Size(sample_shape)

            #samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
            log_mu = self.log_mu[None,:].expand(sample_shape.numel(), *self.log_mu.shape)
            index = torch.nn.functional.gumbel_softmax(log_mu, hard=False)
            action = action.reshape(-1, *action.shape[len(sample_shape):])
            action = (index * action).sum(dim=2)
            action = action.reshape(*sample_shape, *action.shape[1:])
        else:
            index = self.mode.sample(sample_shape, *args, **kwargs) # we store index, there is no gradient here ..

            #return action[torch.arange(len(action), device=action.device), self.index]
            #return batched_index_select(action, 2, self.index)
            sample_shape, batch_size = index.shape[:-1], index.shape[-1]
            d = len(sample_shape)

            action_shape = action.shape[d+1:]
            action = action.reshape(-1, *action_shape) #(sample * batch, mode, action_shape)
            action = action[torch.arange(len(action), device=action.device), index.reshape(-1)]

            action = action.reshape(*sample_shape, batch_size, *action_shape[1:])
        #return action, torch.log(self.probs(action))#self.mode.log_prob(index) # *sample_shape, batch, action_shape
        return action, self.mode.log_prob(index)


    def sample(self, *args, **kwargs):
        #self.index = self.mode.sample(*args)
        #action = self.dist.sample(**kwargs)
        #return action[torch.arange(len(action), device=action.device), self.index]
        #return self.rsample(*args, **kwargs).detach()
        raise NotImplementedError

    def entropy(self, n=1000):
        assert self.log_mu.shape[0] == 1
        #action, _ = self.rsample((n,))
        action = torch.rand(size=(n, 1), device=self.dist.loc.device) * 2 - 1
        #action = self.rsample((n,))[0]
        prob = self.probs(action.detach()).clamp(1e-30)
        entropy = torch.log(prob) # * prob
        return entropy.sum(axis=0)

    def probs(self, action, sum=True):
        action_shape =self.dist.loc.shape[2:]
        action = self.batch_action(action).unsqueeze(-len(action_shape)-1)

        log_prob = self.dist.log_prob(action)

        #assert torch.allclose(log_prob[-2], self.dist.log_prob(action[-2]))
        for _ in action_shape:
            log_prob = log_prob.sum(axis=-1) # sum up over action space ..

        probs_density = torch.exp(log_prob)
        pi = self.mode.probs.expand_as(probs_density)
        #return torch.log()
        return torch.sum(pi * probs_density, axis=-1)

    def get_parameters(self):
        return self.log_mu, self.loc, self.scale

    # values ..
    def REINFORCE(self, values):
        assert not self.gumbel

        log_prob = self.mode.log_prob(self.index)
        assert log_prob.shape == values.shape, f"{log_prob.shape}, {values.shape}"
        return log_prob * values.detach() # we need to maximize log p * R
