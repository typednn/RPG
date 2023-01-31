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

    INPUT_ARGS = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.init(*args, **kwargs)
        
    def init(self, *args, **kwargs):
        self.inp_types = list(map(get_type_from_op_or_type, args))

        if self.INPUT_ARGS is not None:
            assert len(self.INPUT_ARGS) == len(self.inp_types), f"expected {len(self.INPUT_ARGS)} inputs but got {len(self.inp_types)}"
            self.inp_dict = dict(zip(self.INPUT_ARGS, self.inp_types))

        not_Types = False
        for i in self.inp_types:
            if not isinstance(i, Type):
                not_Types = True
            elif not_Types:
                raise TypeError(f"inputs must be a list of Type or Operator then followed by other attributes..")

        self.build_config(**kwargs)
        self.build_modules(*self.inp_types)
        self.type_inference(*self.inp_types)

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

    def _type_inference(self, *inp_types):
        self._inp_type, _, self._oup_type = self.arrow.unify(inp_types)
        return self._oup_type

    def type_inference(self, *inp_types):
        if self.arrow is None:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in inp_types]
            output = self.forward(*shapes)
            self._oup_type = self._get_type_from_output(output, *inp_types)
        else:
            self._oup_type = self._type_inference(*inp_types)

    @property
    def out(self): # out type when input are feed ..
        if not hasattr(self, '_oup_type'):
            raise NotImplementedError(f"the type_inference function is not called for class {self.__class__}")
        return self._oup_type

    def __str__(self) -> str:
        out = super().__str__()
        return out + f"\nOutputType: {self.out}"

    def _get_type_from_output(self, output, *args):
        inp = self.inp_types[0]
        out_shape = inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return out_shape

        
        
class TypedFactory(Operator):
    factory = {}
    def build_modules(self, *args):
        #return super().build_modules(*args)
        from .unification import unify, TypeInferenceFailure
        inp_types = [i for i in self.inp_types if isinstance(i, Type)]
        op = None
        for types, op in self.factory.items():
            try:
                unify(inp_types, types, inp_types)[-1]
            except TypeInferenceFailure:
                continue
            op = op
            break

        if op is None:
            raise TypeError(f"no matching operator for {self.inp_types} in {self.__class__.__name__}\n Factory: {self.factory}")
        
        self.op = op


    @classmethod
    def register(cls, opt, inp_types):
        raise NotImplementedError
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError