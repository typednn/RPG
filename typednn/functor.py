# Functor takes in modules as input and wrap them into a new module.
import torch
from .operator import Operator
from omegaconf import OmegaConf as C


class Functor(Operator):
    # in fact a module dict
    inherit_config = False

    def __init__(
        self,
        *args,
        name=None,
        _trace_history=None,
        **kwargs
    ) -> None:
        super().__init__(*args, name=name, _trace_history=_trace_history, **kwargs)
        self.submodules = dict(zip(self._init_keys, self._init_args))

    def build_modules(self):
        # for i in self.submodules.values():
        #     i.build_modules()
        self.main = torch.nn.ModuleDict({name: k for k, name in self.submodules.items()})
        #for k, name in self.submodules.items():
        #    setattr(self, name, k)

    def reconfig(self, **kwargs):
        return super().reconfig(**kwargs)

    def build_config(self):
        if self.inherit_config:
            self.main.build_config()
        else:
            config = dict(
            )
            self._config = config

            for name, module in self.submodules.items():
                config[name] = module.config
                config[name]['_type'] = module.__class__.__name__
            self._config = C.create(config)


if __name__ == '__main__':
    pass
