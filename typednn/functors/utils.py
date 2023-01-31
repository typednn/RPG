# utility function like concat, stack and so on ..
import numpy as np
from torch import nn
from ..operator import Operator
from ..basetypes import Type
from ..types.tensor import TensorType, Arrow

#class Concat(Operator):
#    def type_inference(self, *args):
#        self._oup_type = args[-1].out
    

# class FlattenBatch(Operator):
#     def __init__(self, modules) -> None:
#         super().__init__(*args, **kwargs)
#     def forward(self, x):
#         dims = self.inp_types[0].data_dims
#         return x.reshape(-1, *x.shape[-dims:])

#     def _type_inference(self, inp_type):
#         batch_shape = inp_type.batch_shape()
#         total = 1 
#         for i in batch_shape:
#             try:
#                 b = int(i)
#                 total *= b
#             except TypeError:
#                 total = Type("X")
#         return inp_type.new(Type("*"), int(np.prod(inp_type.data_shape())))

    
class Flatten(Operator):
    """Flattens its input to a (batched) vector."""
    arrow = 'none'
    def forward(self, x):
        dims = self.inp_types[0].data_dims
        return x.reshape(*x.shape[:-dims], -1)


    def _type_inference(self, inp_type):
        # print(inp_type.data_shape())
        return inp_type.new(*inp_type.batch_shape(), int(np.prod(inp_type.data_shape())), data_dims=1)


class Seq(Operator):
    arrow = 'none'

    def __init__(self, *modules) -> None:
        self.op_list = modules
        super().__init__()

    def build_modules(self, *args):
        from tools.utils import Seq
        self.main = Seq(*self._modules)

    def _type_inference(self, *args, **kwargs):
        return self.op_list[-1].out

    def __str__(self) -> str:
        out = 'Sequential:\n'
        for i in self.op_list:
            out += '  ' + str(i).replace('\n', '\n   ') + '\n'
        out += str(self.out)
        return out

class Linear(Operator):
    #arrow = Arrow(TensorType('...', 'N'), TensorType("X", "X"))
    
    def _type_inference(self, inp_type):
        assert isinstance(inp_type, TensorType)
        assert inp_type.data_dims == 1
        # print(inp_type.data_shape())
        return inp_type.new(*inp_type.batch_shape(), int(np.prod(inp_type.data_shape())))

if __name__ == '__main__':
    pass