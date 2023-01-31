import typing
import numpy as np
import torch
from ..basetypes import Type, VariableArgs, TupleType, Arrow
from .size import SizeType, UIntType


class TensorType(Type):
    def __init__(self, *size, dtype=None, device=None, data_dims=None):
        _size = []
        for i in size:
            if isinstance(i, str):
                if i == '...':
                    i = VariableArgs(i)
                else:
                    i = Type(i)
            _size.append(i)

        size = SizeType(*_size)
        assert isinstance(size, SizeType) # size could be either an size type or a type variable ..
        self.size = size
        self.data_cls = torch.Tensor

        self.dtype = dtype or torch.float32
        self.device = device or 'cuda:0'
        assert data_dims is not None, "data_dims must be specified for batch tensor"
        self.data_dims = data_dims


    def new(self, *size, data_dims=None):
        return TensorType(*size, dtype=self.dtype, device=self.device, data_dims=data_dims or self.data_dims)

    def batch_shape(self):
        return self.size[:-self.data_dims]

    def data_shape(self):
        return tuple([int(i) for i in self.size[-self.data_dims:]])

    @property
    def channel_dim(self):
        C = self.size[-self.data_dims]
        try:
            C = int(C)
        except TypeError as e:
            raise TypeError(str(e) + f"\n    The actual channel is Type {C}")
        return C

    def instance(self, value):
        if not (isinstance(value, self.data_cls)):
            return False
        return self.size.instance(value.shape)

    def children(self):
        return self.size.children()

    def __str__(self):
        #out = self.type_name
        out = 'Tensor(' + ','.join(map(str, self.batch_shape())) + ' : ' + ','.join(map(str, self.data_shape())) + ')'
        if self.dtype != torch.float32:
            out = 'D' + out
        if self.device != 'cuda:0':
            out = 'Cpu' + out
        return out

    def sample(self):
        return torch.randn(*self.size.sample(), device=self.device, dtype=self.dtype)
        

def test():
    N = Type('N')
    M = Type('M')
    shapeA = TensorType(3, N, M)
    shapeB = TensorType(VariableArgs('...'), 4, 5)
    shapeC = TensorType(N, M, VariableArgs('...'))
    arrow = Arrow(shapeA, shapeB, shapeC)
    print(arrow)

    X = TensorType(3, 4, 5)
    B = TensorType(10, 10, 4, 5)

    arrow.test_unify("Tensor(4, 5, 10, 10)", X, B)


    shapeA = TensorType(VariableArgs('...'), 4, 5)
    shapeB = TensorType(N, M, VariableArgs('...'))
    shapeC = TensorType(N, M, VariableArgs('...'))
    arrow = Arrow(shapeA, shapeB, shapeC)

    #print(arrow.unify(X, B))
    arrow.test_unify("error", X, B)

    arrow.test_unify("Tensor(4, 5, 4, 5)", TensorType(4, 5, 4, 5), TensorType(4, 5, VariableArgs('...')))

    arrow.test_unify("Tensor(2, 2, 1)", TensorType(1, 4, 5), TensorType(2, 2, 1))

    # arrow.test_unify("error", TensorType(1, 4, 5), TensorType(2, 2, 2))
        
        
if __name__ == '__main__':
    test()