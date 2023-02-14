import torch
from ..operator import Code, Arrow
from tools.utils import AttrDict
from ..types import AttrType, Type, TensorType, VariableArgs

class KWArgsOp(Code):
    arrow = Arrow(VariableArgs("...", None), AttrType(a=Type("a"), b=Type("b")))

    def forward(self,  a, b, *args):
        return AttrDict(a=a, b=b)


def test():
    inp1 = TensorType(3, 4)
    inp2 = TensorType(3, 4)

    x = KWArgsOp(inp1, x=inp2)
    print(x)
    

    a = inp1.sample()
    b = inp2.sample()

    A = x(a, b)

    error = None
    try:
        x(a)
    except ValueError as e:
        error = e

    if error is None:
        raise Exception("should raise error")

    B = x(a, x=b)

    assert torch.allclose(A.a, B.a)
    assert torch.allclose(A.b, B.b)

    error = None
    try:
        C = x(a, x=b, y=1)
    except ValueError as e:
        error = e
    if error is None:
        raise Exception("should raise error")
        
if __name__ == '__main__':
    test()