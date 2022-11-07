# evaluate the exploration ability of the Gaussian policy
# three modes: 
#   - the maximum entropy exploration with a decaying entropy coef. 
#   - effects of the batch size.
#   - effects of the initial std [we always start from the very large one]. 
import gym
from nn.distributions import Normal
from tools.config import Configurable
import torch


def calc_reward(x):
    action = x
    c = torch.tensor([-1.2, 0.75], device='cuda:0').unsqueeze(1)
    x = x.squeeze().unsqueeze(0).repeat(c.shape[0], 1)
    d = (x - c)
    d[0] *= 6
    e = (0.5 * d ** 2)
    e[0] -= 0.6
    e = e.min(0)[0]
    r = torch.clamp(-e, min=-0.5)
    jump = c[1]
    return r * (action.squeeze() < jump).float() + (action.squeeze() > jump).float() * (-0.5)


class train(Configurable):
    def __init__(self, cfg=None, head=Normal.dc, lr=0.01, batch_size=1024, max_steps=1000000) -> None:
        super().__init__()

        action_space = gym.spaces.Box(-1, 1, (1,))
        self.pi = Normal(action_space, cfg=head)

        params = torch.nn.Parameter(torch.zeros(head.get_input_dim(), requires_grad=True))
        self.optim = torch.optim.Adam(params, lr=lr)

    def train(self):
        for i in range(self._cfg.max_steps):
            a = self.pi.sample()
            r = calc_reward(a)
            loss = -r.mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if i % 1000 == 0:
                print(f'iter {i}: {loss.item()}')

    def draw(self):
        x = torch.tensor()



trainer = train.parse()