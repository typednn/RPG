import torch
from tools.constrained_optim import COptim


x = torch.nn.Parameter(torch.ones(1, requires_grad=True))

def c(x):
    return 0.1 - x

def f(x):
    return (x**2).sum()


optim = COptim(x, 1, weight_penalty=1e-5)

while True:
    optim.optimize(f(x), c(x))
    print(x)