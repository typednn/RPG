# utility function like concat, stack and so on ..
import numpy as np
from torch import nn
from ..operator import Operator
from ..functor import Functor
from ..types.tensor import TensorType, Arrow, TupleType, Type, VariableArgs


class FlattenBatch(Operator):
    def forward(self, x):
        return x.reshape(-1, *self.default_inp_nodes[0].get_type().data_shape().as_int())

    def _type_inference(self, inp_type):
        return inp_type.new(inp_type.batch_shape().total(), *inp_type.data_shape())
    

class Flatten(Operator):
    def forward(self, x):
        dims = self.default_inp_nodes[0].get_type().data_dims
        return x.reshape(*x.shape[:-dims], -1)

    def _type_inference(self, inp_type):
        # print(inp_type.data_shape())
        return inp_type.new(*inp_type.batch_shape(), inp_type.data_shape().total(), data_dims=1)


class Seq(Functor):
    def __init__(self, *modules, **kwargs) -> None:
        self.op_list = modules
        super().__init__(*modules, **kwargs)

    def forward(self, *args, **kwargs):
        out = args
        for k, module in self.main.items():
            out = [module(*out)]
        return out[0] 

    def _input_modules(self):
        return self.op_list[0]

    def _output_modules(self):
        return self.op_list[-1]

    def __str__(self) -> str:
        out = 'Sequential:\n'
        for i in self.op_list:
            out += '  ' + str(i).replace('\n', '\n   ') + '\n'
        out += str(self.out)
        return out


class Concat(Operator):
    def forward(self, *args):
        import torch
        return torch.cat(args, dim=self.dim)

    def _type_inference(self, *args, **kwargs):
        assert len(args) > 1
        out = args[0]
        for i in args[1:]:
            out = out.new(*out.batch_shape(), *out.data_shape().concat(i.data_shape(), self.dim))
        return out


class Tuple(Operator):
    arrow = Arrow(VariableArgs('...', None), VariableArgs('...', None))
    def forward(self, *args):
        return args

    def _type_inference(self, *input_types):
        from ..basetypes import TupleType, Type
        return TupleType(*input_types)


class Dict(Operator):
    def __init__(self, *args, name=None, _trace_history=None, **kwargs) -> None:
        if len(args) > 0:
            assert len(args) == 1 and isinstance(args[0], dict)
            assert len(kwargs) == 0
            kwargs = args[0]
        super().__init__(*args, name=name, _trace_history=_trace_history, **kwargs)

    def forward(self, *args):
        from tools.utils import AttrDict
        return AttrDict(**{k:v for k, v in zip(self._init_keys, args)})

    def _type_inference(self, *input_types):
        from ..types import AttrType
        return AttrType(**{k:input_types[i] for i, k in enumerate(self._init_keys)})


class Linear(Operator):
    @classmethod
    def _new_config(cls):
        return dict(
            dim=256,
        )

    def build_modules(self, inp_type):
        self.myassert(isinstance(inp_type, TensorType), "Linear only support TensorType but got: " + str(inp_type))
        self.myassert(inp_type.data_dims == 1, "Linear only support 1D TensorType but got: " + str(inp_type))
        self.main = nn.Linear(inp_type.channel_dim, self.config.dim).to(inp_type.device)
    
    def _type_inference(self, inp_type):
        return inp_type.new(*inp_type.batch_shape(), self.config.dim)

    def forward(self, *args, **kwargs):
        return self.main(*args, **kwargs)


if __name__ == '__main__':
    pass