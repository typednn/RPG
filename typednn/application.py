# Handeling the calling of the operator
from .node import Node
from omegaconf import OmegaConf as C
from .basetypes import Type


def process_args_kwargs(*args, **kwargs):
    args = list(args)
    keys = [f'args{i}' for i in range(len(args))]
    for k, v in kwargs.items():
        args.append(v)
        keys.append(k)
    return keys, args


class CallNode(Node):
    def __init__(self, op, *args, key=None, **kwargs):
        from .operator import ArrowNode
        op: ArrowNode = op
        self.op = op
        self.input_keys, self.input_nodes = process_args_kwargs(*args, **kwargs)
        self.input_nodes = list(map(Node.from_val, self.input_nodes))
        self.trace_key = key or op.__class__.__name__
        super().__init__(None) 

        # context is None
        self._meta_type = op._type_inference(*[
            i._meta_type for i in self.input_nodes], context=None)
        self.sync()

    def sync(self):
        self.op._input_keys = self.input_keys
        self.op._input_nodes = self.input_nodes

    def get_parents(self):
        return [self.op] + list(self.input_nodes)

    def print_line(self):
        return self.op._name + '(' +  ', '.join([j._name if j._name else str(j) for j in self.input_nodes]) + ')'

    def _get_type(self, op_type, *inp_types, context):
        self.sync()
        return self.op._type_inference(*inp_types, context=context)

    @property
    def config(self):
        return self.op.config

    @property
    def pretty_config(self):
        return self.op.pretty_config

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

    def _get_evaluate(self, op, *args, context=None, **kwargs):
        from .context import Context
        context: Context = context

        inps = self.match_input(*args, **kwargs)
        for node, input in zip(self.input_nodes, inps):
            type = context.type[node]
            if isinstance(type, Type) and type.instance(input) is None:
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                self.myassert(False, f"input {input} does not match the required input type {type} {info}", TypeError)

        self.sync()
        out = op(*inps)

        out_type = context.type[self]
        assert out_type.instance(out) is not None, f"output {out} does not match the required output type {out_type}"
        return out

    def reuse(self, *args, **kwargs):
        return self.op.reuse(*args, key=self._name, **kwargs)

    def eval(self, *args, **kwargs):
        return self._get_evaluate(
            self.default_context.evaluate[self.op], *args, context=self.default_context, **kwargs)


class DataNode(CallNode):
    def __len__(self):
        self.init()
        return len(self.op)