import typing
from torch.nn import Module
from .basetypes import Arrow
# pytorch module, but typed
# shapes that we support:
#   - (num1, num2, num3)
#   - one ellipse (num1, ..., num2, num3)
#   - (num1, num2, IMG_TYPE)
#   - each num can be '?' that matches any other shapes or a type symbol

#   shape can be viewed as a way to compose type class 
# we don't support broadcast ..

# operator can take multiple types .. but during type inference, the results must be the same ..

def get_type_from_op_or_type(op_or_type):
    if isinstance(op_or_type, Operator):
        return op_or_type.out
    else:
        return op_or_type


class Operator(Module):
    arrow: typing.Optional[Arrow] = None # TYPE annotation of the forward funciton

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def register_types(self, *args, **kwargs):
        assert len(kwargs) == 0, NotImplementedError
        assert self.arrow is not None, f"arrow is not defined for class {self.__class__}"
        self._inp_type, _, self._oup_type = self.arrow.unify(
            *map(get_type_from_op_or_type, args)
        )

    @property
    def out(self): # out type when input are feed ..
        if not hasattr(self, '_oup_type'):
            raise NotImplementedError(f"the register_types function is not called for class {self.__class__}")
        return self._oup_type

    def _get_type(self):
        if not hasattr(self, '_type'):
            self._type = self.get_type()
        return self._type


    def __str__(self) -> str:
        out = super().__str__()
        return out + f" {self.arrow}: {self.out}"