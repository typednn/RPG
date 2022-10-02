import torch
from torch.distributions import Normal

def main():

    mean = torch.nn.Parameter(torch.tensor([0.]))
    logstd = torch.nn.Parameter(torch.tensor([0.]))

    optim = torch.optim.Adam([mean, logstd], lr=0.1)

    for i in range(100):
        dist = Normal(mean, logstd.exp())

        a = dist.rsample()

        logp = dist.log_prob(a.detach())

        optim.zero_grad()

        loss = logp * ((a-1.)**2).detach()

        loss.backward()
        print(mean.grad)

        optim.step()
    print(mean)

if __name__ == '__main__':
    main()
