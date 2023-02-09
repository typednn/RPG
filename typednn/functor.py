# Functor takes in modules as input and wrap them into a new module.
import torch
from .operator import Operator
from omegaconf import OmegaConf as C


class Functor(Operator):
    def __init__(
        self,
        *args, name=None,
        _trace_history=None,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            name=name,
            _trace_history=_trace_history,
            **kwargs
        )
        self.submodules = dict(
            zip(self._init_keys, self._init_args)
        )
        self._init_keys = []
        self._init_args = []

    def clear(self):
        super().clear()
        self._default_inp_nodes = [] # the default inp nodes will be [] without initliaze the submodules

    def _get_arrow(self):
        raise NotImplementedError

    def _type_inference(self, *input_types):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def build_modules(self):
        for name, module in self.submodules.items():
            module.init() # initlaize the module
        self.main = torch.nn.ModuleDict(self.submodules)
        self._get_arrow()

    def reconfig(self, **kwargs):
        #return super().reconfig(**kwargs)
        outs = {}
        for k, v in kwargs.items():
            if k in self.submodules:
                self.submodules[k].reconfig(**v)
            else:
                outs[k] = v
        super().reconfig(**outs)

    def build_config(self):
        config = C.merge(self.default_config(), self._init_kwargs)
        for name, module in self.submodules.items():
            config[name] = module.config
            config[name]['_type'] = module.__class__.__name__
        self._config = C.create(config)


if __name__ == '__main__':
    pass
