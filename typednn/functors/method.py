"""
Inheritance or override parents' methods are a very common issue.

By default every sub type class will create a new version of neural network for each method (Notice that this is not for the type instance). 
    - for example we can create a neural network for Integer, that does not mean we will create it for int(0), int(1) and so on ..
If and only if the method is not defined in the current class, it will be inherited from the parent class.
"""

import inspect
from ..abstraction import Function
from ..basetypes import AttrType
from .funcdef import asfunc
from .pyfunc import torchop
from ..operator import Code


def get_class_that_defined_method(cls, methname):
    for cls in inspect.getmro(cls):
        if methname in cls.__dict__: 
            return cls
    return None


class Method:
    def __init__(self, name, func) -> None:
        super().__init__()
        self.name = name
        self.func = func
        self.annotation = func.__annotations__

        self.factory = {}

    def __call__(self, *args, **kwargs):
        # TODO: by default we will
        inp_type = args[0]
        inp_type_class = str(inp_type.__class__.__name__)

        if inp_type_class not in self.factory:
            annotation = {'_self': inp_type, **self.func.__annotations__}
            func = self.factory[inp_type_class] = asfunc(self.func, annotation=annotation)
        else:
            func = self.factory[inp_type_class]
        return func(*args, **kwargs)


def asmethod(func) -> "Code":
    name = func.__name__
    method = Method(name, func)

    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)

    wrapper.method = method
    return wrapper


class Class(AttrType):
    def new(self, *args, **kwds):
        raise NotImplementedError("change the way")
        out = self.sample()
        out.update(kwds)
        return out


def test():

    from ..types import TensorType, MLP, PointDict
    class NewType(AttrType):
        a: TensorType('B', 'N')

        @asmethod
        def encode(self):
            return MLP(self.a)


    class NewType2(NewType):
        a: TensorType("B", 100)

    class NewType3(NewType2):
        a: TensorType(300, 100)

        @asmethod
        def encode(self):
            return MLP(self.a, out_dim=10)

    class NewType4(NewType):
        a: TensorType(300, 100)

        @asmethod
        def encode(self):
            return MLP(self.a, out_dim=10)

    new_type1 = NewType()
    new_type2 = NewType2()
    new_type3 = NewType2()
    new_type4 = NewType3()
    assert new_type1.encode.method is new_type2.encode.method
    assert new_type3.encode.method is new_type2.encode.method and new_type3.encode.method is NewType.encode.method
    assert NewType3.encode.method is not NewType2.encode.method

    inp = NewType2(a=TensorType(512, 100)).sample()

    #print(inp.encode())
    # new_type1.encode()

    out = new_type2.encode()
    inp = NewType3().sample()
    x1 = out.eval(inp)
    print(x1.shape)

    out2 = new_type3.encode()
    x2 = out2.eval(inp)
    import torch
    assert torch.allclose(x1, x2)
    print(x2.shape)

    x3 = new_type4.encode().eval(inp)
    assert x3.shape[-1] == 10

    
if __name__ == '__main__':
    test()