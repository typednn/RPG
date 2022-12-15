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

    def make_discriminator(self):
        raise NotImplementedError

    def relabel(self):
        raise NotImplementedError

    @property
    def learn(self):
        return True

        
class Categorical(HiddenSpace):
    def __init__(
        self, cfg=None, n=1,
    ) -> None:
        super().__init__()
        self.n = n 
        from gym.spaces import Discrete as D
        self._space = D(self.n)

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

    def make_info_head(self, cfg=None):
        #discrete = dict(TYPE='Discrete', epsilon=self._cfg.epsilon)  # 0.2 epsilon
        default = Discrete.gdc(epsilon=0.2)
        cfg = CN(cfg)
        cfg = merge_a_into_b(cfg, default)
        return Discrete(self.space, cfg=cfg)


class Gaussian(HiddenSpace):
    def __init__(
        self, cfg=None, dim=6,
    ) -> None:
        super().__init__()
        self._dim = dim 
        self._space = Box(-1, 1, (dim,))

    @property
    def dim(self):
        return self._dim

    @property
    def space(self):
        return self._space

    def tokenize(self, z):
        return z

    def make_policy_head(self, cfg=None):
        default = Normal.gdc(linear=True, std_scale=1., std_mode='fix_no_grad', nocenter=True, squash=False)
        return Normal(self.space, cfg=merge_a_into_b(CN(cfg), default))

    def make_info_head(self, cfg=None):
        #discrete = dict(TYPE='Discrete', epsilon=self._cfg.epsilon)  # 0.2 epsilon
        continuous = Normal.gdc(linear=True, std_mode='fix_no_grad', std_scale=0.3989)
        cfg = CN(cfg)
        cfg = merge_a_into_b(cfg, continuous)
        return Normal(self.space, cfg=cfg)