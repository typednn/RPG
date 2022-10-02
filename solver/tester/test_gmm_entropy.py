import torch
from torch import nn
import numpy as np
from torch.distributions import Normal
from solver.mixture_of_guassian import GMMAction

def f(x):
    x = torch.clamp(x, -2., 2.)
    assert isinstance(x, torch.Tensor)
    flag = x < 0.3
    return (2.-(x + 0.5)**2 * 4)*flag +  (4. - (x-0.8)**2 * 0.6) * (1-flag.float())

def main():

    n = 3
    log_mu = torch.tensor(np.array([0.,] * n))[None, :]
    loc = torch.tensor(np.array([0.0, ] * n))[None, :]
    log_std = torch.tensor(np.log(np.array([0.1] * n)))[None, :]

    params = [log_mu, loc]
    params += [log_std]
    params = [nn.Parameter(i) for i in params]
    #params += [log_std]
    log_mu, loc, log_std = params

    optim = torch.optim.Adam(params, lr=0.01)
    T = 3000

    for i in range(T):
        dist = GMMAction(log_mu, loc, log_std.exp())

        a, logp = dist.rsample()

        #logp = dist.log_prob(a.detach())

        optim.zero_grad()

        #r = ((a-1.)**2)
        neg_r = -f(a)

        loss = logp * neg_r.detach() + neg_r  #- dist.entropy(10000)

        #log_std = torch.tensor(np.log(np.array([0.2, 0.2]) * (i/1000.)))[None, :]
        #en = dist.entropy() * 0.001 #max(1-(i/T) *  2, 0.) #* 20. #0.3
        en = dist.entropy() * max(1-(i/T) *  2, 0.) * 0.01
        loss -= en
        #if i == 2900:
        #    optim = torch.optim.Adam(params, lr=0.03)

        loss.backward()

        optim.step()

    print(loc, torch.softmax(log_mu, -1))
    print(log_std.exp())

if __name__ == '__main__':
    main()
