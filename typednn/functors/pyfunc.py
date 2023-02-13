# define pytorch wrapper for TensorTypes; not recommended actually ..
from ..types import TensorType, VariableArgs, Arrow
from ..code import Code
from ..application import CallNode


# something like a pytorch wrapper 
class PyOp(Code):
    func = None
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    
    def __init__(self, function) -> None:
        super().__init__()
        self._name = function.__name__
        self.func = function

        annotation = function.__annotations__
        assert 'return' in annotation, 'return type annotation is required'

        self.arrow = Arrow(**annotation)

    def __call__(self, *args, **kwargs):
        return self.reuse(*args, key=self._name, **kwargs)

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def torchop(func):
    return PyOp(function=func)

    
def test():
    @torchop
    def add(a: TensorType(VariableArgs('...')), b: TensorType(VariableArgs('...'))) -> TensorType(VariableArgs('...')):
        return a + b

    x = TensorType(1, 2, 3)

    out = add(x, x)

    y = TensorType(VariableArgs('...'))
    out = add(y, y)

    import torch
    a = torch.tensor([2,3,4])
    b = torch.tensor([4,2,4])
    p = out.eval(a, b)
    print(p)

    
if __name__ == '__main__':
    test()