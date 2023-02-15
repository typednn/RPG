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
    def __init__(self, op, *args, key=None, reconfig=True, **kwargs):
        from .operator import ArrowNode
        from .context import get_context

        op: ArrowNode = op
        self.input_keys, self.input_nodes, init_kwargs = process_args_kwargs(op.default_config(), *args, **kwargs)
        self.input_nodes =list(map(Node.from_val, self.input_nodes))
        if reconfig:
            op.reconfig(**init_kwargs)
        else:
            assert len(init_kwargs) == 0

        self.op = op
        self.trace_key = key or self.op.__class__.__name__
        meta_type = op._type_inference(*[i._meta_type for i in self.input_nodes], context=get_context()) # context is None
        super().__init__(meta_type)

    def get_parents(self):
        return [self.op] + list(self.input_nodes)

    def print_line(self):
        return self.op._name + '(' +  ', '.join([j._name if j._name else str(j) for j in self.input_nodes]) + ')'

    def _get_type(self, op_type, *inp_types, context):
        return self.op._type_inference(*inp_types, context)

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
        inps = self.match_input(*args, **kwargs)
        for node, input in zip(self.input_nodes, inps):
            type = context.type[node]
            if isinstance(type, Type) and type.instance(input) is None:
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                self.myassert(False, f"input {input} does not match the required input type {type} {info}", TypeError)
        out = op(*inps)
        out_type = context[self]
        assert out_type.instance(out) is not None, f"output {out} does not match the required output type {out_type}"
        return out

    def reuse(self, *args, **kwargs):
        return self.op.reuse(*args, key=self._name, **kwargs)

    def eval(self, *args, **kwargs):
        print('eval', self.context.evaluate[self.op])
        exit(0)
        return self._get_evaluate(self.context.evaluate[self.op], *args, context=self.context, **kwargs)


class DataNode(CallNode):
    def __len__(self):
        self.init()
        return len(self.op)