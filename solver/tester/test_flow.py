import normflow as nf
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os

#base = nf.distributions.base.DiagGaussian(2, trainable=False)
base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
#spline_transform = T.spline_coupling(2, count_bins=16)
spline_transform = T.spline_coupling(2, count_bins=16)
flow_dist = dist.TransformedDistribution(base_dist, [spline_transform, T.TanhTransform()])


def f(x):
    x = torch.clamp(x, -2., 2.)
    assert isinstance(x, torch.Tensor)
    flag = x > 0.1
    return (4.-(x - 0.8)**2 * 4)*flag +  (10. - (x+0.5)**2 * 0.4) * (1-flag.float())


import torch
optim = torch.optim.Adam(spline_transform.parameters(), lr=0.001)
import tqdm
T = 2000

for i in tqdm.trange(T):
    #a, logp = model.sample(200)
    a = flow_dist.sample((100,))
    logp = flow_dist.log_prob(a.detach())
    #if torch.isnan(logp).any():
    #    print('nan')
    #    continue
    optim.zero_grad()

    #r = ((a-1.)**2)
    r = f(a[:, 0]) - a[:, 1].abs() * 10

    #loss = ((-logp * r.detach()).mean() )# - r.mean())  #- dist.entropy(10000)
    loss = ((-logp * r.detach()).mean() - r.mean())  #- dist.entropy(10000)
    #loss = 0

    #loss = loss.mean()
    #log_std = torch.tensor(np.log(np.array([0.2, 0.2]) * (i/1000.)))[None, :]
    x = torch.rand((100,2), device=a.device) * 2 - 1
    x[:, 1] *= 1.
    en = flow_dist.log_prob(x).mean() * 2.
    #en = -logp.mean()
    loss -= en
    #if i == 2900:
    #    optim = torch.optim.Adam(params, lr=0.03)

    loss.backward()
    optim.step()

    if i % 100== 0:
        x = flow_dist.sample((10000,))
        plt.clf()
        plt.hist(x[:, 0].detach().cpu().numpy(), bins=30)
        plt.savefig('x.png')
        x = flow_dist.sample((1,))
        print(x, f(x[:, 0]), f(x[:, 0]*0+0.9))
    if torch.isnan(loss):
        break