# define pytorch wrapper for TensorTypes; not recommended actually ..
from ..types import TensorType, VariableArgs, Arrow, Type
from ..operator import ArrowNode
from ..application import CallNode


# something like a pytorch wrapper 
class PyOp(ArrowNode):
    func = None
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    
    def __init__(self, function, annotation) -> None:
        self.func = function

        import copy
        if annotation is None:
            annotation = function.__annotations__
        assert 'return' in annotation, 'return type annotation is required'
        self.arrow = Arrow(**annotation)
        super().__init__()
        self._name = function.__name__

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _type_inference(self, *input_types, context) :
        out = self.arrow.unify(*input_types)[-1]
        return out

    def _get_type(self, *args, context):
        return super()._get_type(*args, context=context)

    def __call__(self, *args, **kwargs):
        return CallNode(self, *args, key=self._name, **kwargs)

    def _get_evaluate(self, *parents_callable, context=None):
        return self.func

    def __str__(self) -> str:
        return super().__str__() + f'({self.func})'


def torchop(func, annotation=None):
    # TODO: allow to specify configs here
    return PyOp(function=func, annotation=annotation)

    
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