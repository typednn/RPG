import torch
from tools.config import Configurable, as_builder, merge_a_into_b, CN
from nn.distributions.discrete import Discrete
from nn.distributions.normal import Normal
from gym.spaces import Box


@as_builder
class HiddenSpace(Configurable):
    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)

    @property
    def dim(self):
        pass

    def tokenize(self, z):
        pass

    def make_policy_head(self, cfg=None):
        raise NotImplementedError

    def get_input_dim(self):
        # return parameters for identify the hidden variables in maximizing the mutual information
        raise NotImplementedError

    def likelihood(self, inp, z, timestep):
        raise NotImplementedError

    def reward(self, inp, z, timestep):
        return self.likelihood(inp, z, timestep)

    def relabel(self):
        raise NotImplementedError

    @property
    def learn(self):
        return True

    def callback(self, trainer):
        # used to 
        pass

        
class Categorical(HiddenSpace):
    def __init__(
        self, cfg=None, n=1,
        head=Discrete.gdc(epsilon=0.2),
    ) -> None:
        super().__init__()
        self.n = n 
        from gym.spaces import Discrete as D
        self._space = D(self.n)
        self.head = Discrete(self.space, cfg=head)

    @property
    def dim(self):
        return self.n

    @property
    def space(self):
        return self._space

    @property
    def learn(self):
        return self.n > 1
        

    def tokenize(self, z):
        return torch.nn.functional.one_hot(z, self.n).float()

    def make_policy_head(self, cfg=None):
        return Discrete(self.space, cfg=cfg)

    def get_input_dim(self):
        return self.n

    def likelihood(self, inp, z, timestep):
        return self.head(inp).log_prob(z)


class Gaussian(HiddenSpace):
    def __init__(
        self, cfg=None, dim=6,
        head = Normal.gdc(linear=True, std_mode='fix_no_grad', std_scale=0.3989)
    ) -> None:
        super().__init__()
        self._dim = dim 
        self._space = Box(-1, 1, (dim,))
        self.head = Normal(self.space, head).cuda()

    @property
    def dim(self):
        return self._dim

    @property
    def space(self):
        return self._space

    def tokenize(self, z):
        return z

    def get_input_dim(self):
        return self.head.get_input_dim()

    def make_policy_head(self, cfg=None):
        default = Normal.gdc(linear=True, std_scale=1., std_mode='fix_no_grad', nocenter=True, squash=False)
        return Normal(self.space, cfg=merge_a_into_b(CN(cfg), default))

    def likelihood(self, inp, z, timestep):
        return self.head(inp).log_prob(z)