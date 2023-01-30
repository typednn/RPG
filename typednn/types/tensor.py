import typing
import torch
from ..basetypes import Type, VariableArgs, TupleType, Arrow


class UIntType(Type):
    def __init__(self, a):
        self.a = a
    def __str__(self):
        return str(self.a)
    def instance(self, value):
        return isinstance(value, int) and value == self.a 

    def __repr__(self):
        return "Unit(" + str(self.a) + ")"

class SizeType(TupleType):
    def __init__(self, *size: typing.List[Type]):
        self.dot = None # (int, ..., int) something like this ..
        self.size = []
        for i in size:
            if isinstance(i, int):
                i = UIntType(i)
            else:
                assert i.match_many() or isinstance(i, UIntType) or i.is_type_variable, f"{i} of {type(i)}"
            self.size.append(i)
        super().__init__(*self.size)

    def __str__(self):
        return '(' + ', '.join(map(str, self.size)) + ')'

    def __repr__(self):
        return 'Size'+self.__str__()


class TensorType(Type):
    def __init__(self, *size):
        size = SizeType(*size)
        assert isinstance(size, SizeType) # size could be either an size type or a type variable ..
        self.size = size
        self.data_cls = torch.Tensor
        self.type_name = 'Tensor' + str(size)

    def instance(self, value):
        if not (isinstance(value, self.data_cls)):
            return False
        return self.size.instance(value.shape)

    def children(self):
        return self.size.children()

    def __str__(self):
        return self.type_name
        

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