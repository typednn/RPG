import typing
from omegaconf import OmegaConf as C
from torch.nn import Module
from .basetypes import Arrow, Type


def get_type_from_op_or_type(op_or_type):
    if isinstance(op_or_type, Operator):
        return op_or_type.out
    else:
        return op_or_type


class OptBase:
    def default_config(self) -> C:
        return C.create()

class Operator(Module, OptBase):
    INFER_SHAPE_FROM_MODULE = False
    arrow: typing.Optional[Arrow] = None # TYPE annotation of the forward funciton

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.init(*args, **kwargs)
        
    def init(self, *args, **kwargs):
        self.build_config(**kwargs)
        self.build_modules(*args)
        self.type_inference(*args)

    def build_modules(self, *args):
        self.main = None
    
    @classmethod
    def _new_config(cls)->C:
        return C.create()

    @classmethod
    def default_config(cls) -> C:
        return C.merge(
            super().default_config(cls),
            cls._new_config()
        )

    def forward(self, *args, **kwargs):
        assert self.main is not None, "please either override this function or set the module of the class"
        return self.main(*args, **kwargs)

    def build_config(self, **kwargs) -> C:
        self.config = C.merge(self.default_config(), C.create(kwargs))

    def _get_type_from_output(self, output, *args):
        raise NotImplementedError("please either override this function or set the arrow attribute of the class")

    def type_inference(self, *args):
        if self.arrow is None:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in args]
            output = self.forward(*shapes)
            self._oup_type = self._get_type_from_output(output, *args)
        else:
            self._inp_type, _, self._oup_type = self.arrow.unify(
                *map(get_type_from_op_or_type, args)
            )

    @property
    def out(self): # out type when input are feed ..
        if not hasattr(self, '_oup_type'):
            raise NotImplementedError(f"the register_types function is not called for class {self.__class__}")
        return self._oup_type

    def __str__(self) -> str:
        out = super().__str__()
        return out + f"\nOutputType: {self.out}"

        
        
class TypeFactory(Operator):
    @classmethod
    def register(cls, name, type):
        raise NotImplementedError