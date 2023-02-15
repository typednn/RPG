# utility function like concat, stack and so on ..
import numpy as np
from torch import nn
from ..operator import Code
from ..types.tensor import TensorType, Arrow, TupleType, Type, VariableArgs


class FlattenBatch(Code):
    def forward(self, x):
        input_nodes = self._context.get_inputs()[0]
        return x.reshape(-1, *input_nodes[0].get_type().data_shape().as_int())

    def _type_inference(self, inp_type, context):
        return inp_type.new(inp_type.batch_shape().total(), *inp_type.data_shape())
    

class Flatten(Code):
    def forward(self, x):
        input_nodes = self._context.get_inputs()[0]

        dims = input_nodes[0].get_type().data_dims
        return x.reshape(*x.shape[:-dims], -1)

    def _type_inference(self, inp_type, context):
        # print(inp_type.data_shape())
        return inp_type.new(*inp_type.batch_shape(), inp_type.data_shape().total(), data_dims=1)


class Concat(Code):
    def forward(self, *args):
        import torch
        return torch.cat(args, dim=self.dim)

    def _type_inference(self, *args, context):
        assert len(args) > 1
        out = args[0]
        for i in args[1:]:
            out = out.new(*out.batch_shape(), *out.data_shape().concat(i.data_shape(), self.dim))
        return out


class Tuple(Code):
    arrow = Arrow(VariableArgs('...', None), VariableArgs('...', None))
    def forward(self, *args):
        return args

    def _type_inference(self, *input_types, context):
        from ..basetypes import TupleType, Type
        return TupleType(*input_types)


class Dict(Code):
    def forward(self, *args):
        from ..attrdict import AttrDict
        return AttrDict(**{k:v for k, v in zip(self._input_keys, args)})

    def _type_inference(self, *input_types, context):
        from ..types import AttrType
        return AttrType(**{k:input_types[i] for i, k in enumerate(self._input_keys)})


class Linear(Code):
    @classmethod
    def _new_config(cls):
        return dict(
            dim=256,
        )

    def build_model(self, inp_type):
        assert isinstance(inp_type, TensorType), "Linear only support TensorType but got: " + str(inp_type)
        assert inp_type.data_dims == 1, "Linear only support 1D TensorType but got: " + str(inp_type)
        return nn.Linear(inp_type.channel_dim, self.config.dim).to(inp_type.device)
    
    def _type_inference(self, inp_type, context):
        return inp_type.new(*inp_type.batch_shape(), self.config.dim)


if __name__ == '__main__':
    pass