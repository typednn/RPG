# Handeling the calling of the operator
import inspect
from .node import Node
from omegaconf import OmegaConf as C
from .basetypes import Type



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

class CallNode(Node):
    def __init__(self, code, *args, caller_key=None, **kwargs):
        from .code import Code
        code: Code = code
        self.input_keys, self.input_nodes, init_kwargs = process_args_kwargs(
            code.default_config(), *args, **kwargs)

        self.input_nodes =list(map(Node.from_val, self.input_nodes))
        code.reconfig(**init_kwargs)

        self.code = code
        self.trace_key = self.code.__class__.__name__
        meta_type = code.type_inference(*[i._meta_type for i in self.input_nodes])
        super().__init__(meta_type)


    def match_input(self, *args, **kwargs):
        inps = list(args)
        for i in range(len(args), len(self.input_nodes)):
            if self.input_keys[i] is None:
                raise ValueError(f"missing input {i}")
            if self.input_keys[i] in kwargs:
                inps.append(kwargs.pop(self.input_keys[i]))
            else:
                raise ValueError(f"missing input {self.input_keys[i]}")
        if len(kwargs) > 0:
            raise ValueError(f"extra inputs {kwargs.keys()}")
        return inps

    def get_parents(self):
        return self.input_nodes

    def print_line(self):
        return self.code._name + '(' +  ', '.join([j._name if j._name else str(j) for j in self.input_nodes]) + ')'

    def _evaluate(self, context):
        for i in self.input_nodes:
            context[i] = i.evaluate(context)
        return self(*[context[i] for i in self.input_nodes])

    def _get_type(self, context):
        if context is None or '_do_not_init' not in context:
            self.code.init(*self.input_nodes) # initailize the operator if needed 
            assert self.code._initialized
        inp_types = [i.get_type(context) for i in self.input_nodes]
        
        out = self.code.type_inference(*inp_types)
        # print(self.op, self.op._initialized, out)
        return out
        
    def __call__(self, *args, **kwargs):
        inps = self.match_input(*args, **kwargs)
        self.code.init(*self.input_nodes) # initalize using the input nodes

        for node, input in zip(self.input_nodes, inps):
            type = node.get_type()
            if isinstance(type, Type) and type.instance(input) is None:
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                #raise Exception("type mismatch..; we need a better way to raise the error.")
                self.myassert(False, f"input {input} does not match the required input type {type} {info}", TypeError)

        self.code.set_input_nodes(*self.input_nodes)
        out = self.code.forward(*inps)
        out_type = self.get_type()
        assert out_type.instance(out) is not None, f"output {out} does not match the required output type {out_type}"
        return out

    def reuse(self, *args, **kwargs):
        return self.code.reuse(*args, **kwargs)