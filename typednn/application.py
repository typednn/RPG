# Handeling the calling of the operator
import inspect
from .node import Node
from omegaconf import OmegaConf as C


def get_left_value(frame):
    frame = frame[0]
    return inspect.getframeinfo(frame).code_context[0].strip().split("=")[0].strip()



def process_args_kwargs(config, *args, **kwargs):
    config_args = {}

    args = list(args)
    keys = [f'args{i}' for i in range(len(args))]

    for k, v in kwargs.items():
        if k in config:
            config_args[k] = v
        else:
            args.append(v)
            keys.append(k)
    return keys, args, C.create(config_args)


class CallNode(Node): # App in the type system.. calling an function..
    # def __init__(self, meta_type, module, input_nodes, **kwargs) -> None:
    #     super().__init__(meta_type, **kwargs)
    #     from .operator import Operator
    #     self.module: Operator = module
    #     self.input_nodes = input_nodes

    def __init__(self, op, *args, **kwargs):
        from .operator import Operator
        self.op: Operator = op
        raise NotImplementedError


    def match_input(self, *args, **kwargs):
        inps = list(args)
        for i in range(len(args), len(self._init_args)):
            if self._init_keys[i] is None:
                raise ValueError(f"missing input {i}")
            if self._init_keys[i] in kwargs:
                inps.append(kwargs.pop(self._init_keys[i]))
            else:
                raise ValueError(f"missing input {self._init_keys[i]}")
        if len(kwargs) > 0:
            raise ValueError(f"extra inputs {kwargs.keys()}")
        return inps, kwargs


    def get_parents(self):
        
        return self.input_nodes

    def print_line(self):
        return self.module._name + '(' +  ', '.join([j._name if j._name else str(j) for j in self.input_nodes]) + ')'

    def _evaluate(self, context):
        for i in self.input_nodes:
            context[i] = i.evaluate(context)
        return self.module(*[context[i] for i in self.input_nodes])

    def _get_type(self):
        return self.module.get_output_type_by_input(*self.input_nodes, force_init=True)



def as_caller(cls):
    #raise NotImplementedError
    def callnode(*args, **kwargs):
        op = cls()
        return CallNode(op, *args, **kwargs)
    return callnode