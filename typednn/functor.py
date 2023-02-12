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

    def _input_modules(self):
        raise NotImplementedError
    
    def _output_modules(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def build_modules(self):
        for name, module in self.submodules.items():
            module.init() # initlaize the module
        self.main = torch.nn.ModuleDict(self.submodules)
        self._get_arrow()

    def _get_arrow(self):
        from .functors import Tuple, Arrow

        input_modules = self._input_modules()
        if isinstance(input_modules, Operator):
            input_modules = [input_modules]
        for i in input_modules:
            self._default_inp_nodes += list(i.default_inp_nodes)

        output_modules = self._output_modules()
        if isinstance(output_modules, list) or isinstance(output_modules, tuple):
            output_nodes = []
            for i in output_modules:
                output_nodes.append(i.get_output())
            output_node = Tuple(*output_nodes)
        else:
            output_node = output_modules.get_output()
        self.arrow = Arrow(*[i.get_type() for i in self._default_inp_nodes], output_node.get_type())

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

    def __call__(self, *args, **kwargs):
        # for functor, the default input nodes is defined here actually 
        return super().__call__(*args, **kwargs)