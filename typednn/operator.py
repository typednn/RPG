# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import torch
from torch import nn
from omegaconf import OmegaConf as C
from torch.nn import Module
from .basetypes import Arrow, Type, VariableArgs
from .node import Node, CallNode
from .utils import frame_assert
from .unification import unify
from typing import Mapping, Any


class OptBase(Module):
    @classmethod
    def default_config(cls) -> C:
        return C.create()

OPID = 0

class Code(OptBase):
    INFER_SHAPE_BY_FORWARD=False
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton
    NODE_MAP=CallNode

    #N_OUTPUT=None

    def __init__(self) -> None: 
        super().__init__()
        self._name = self.__class__.__name__
        global OPID
        self._id = OPID
        OPID += 1
        self._init_kwargs = C.create()
        self.clear()

        self._input_nodes = None

    def set_input_nodes(self, *input_nodes, keys=None):
        self._input_nodes = input_nodes
        self._input_keys = keys

    def clone(self, shallow=True):
        # copy the operator
        raise NotImplementedError
        if shallow:
            assert not self._initialized, "Can't clone an initialized operator"
        new: Code = self.__class__.__new__(self.__class__)
        new.__init__(self._name)
        new.reconfig(**self._init_kwargs)
        return new

    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        if self._initialized:
            #import logging
            #logging.warning(f"reconfiguring a module {self._name} that has already been initialized, this is not recommended")
            raise NotImplementedError("Can't config an initalized operator.")
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self.clear()

    def clear(self):
        self._initialized = False
        self._config = None


    def init(self, *inp_types):
        if not self._initialized:
            self._initialized = True
            try:
                self.main
                has_main = True
            except AttributeError:
                has_main = False
            if not has_main:
                self.build_modules(*inp_types)


    def type_inference(self, *input_types) -> Type:
        return self._type_inference(*input_types)


    def _type_inference(self, *input_types) -> Type:
        from .unification import TypeInferenceFailure
        if self._initialized and self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            inp = input_types[0]
            return inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        _, _, _out_type = self.arrow.unify(*input_types)
        return _out_type

        # error = None
        # try:
        # except TypeInferenceFailure as e:
        #     error = e
        # self.myassert(error is None, f"cannot infer the output type of {self._name} with input {input_types}.\n{error}", error.__class__)

    def forward(self, *args, **kwargs):
        return self.main(*args, **kwargs)
        # assert self.main is not None, "please either override this function or set the module of the class"
        # try:
        # except Exception as e:
        #     import traceback
        #     tb = traceback.format_exc()
        #     self.myassert(False,  f"error in forward function of {self.__class__}:\n{e} with\n {str(tb)}", e.__class__)
            
        
    """ config system """
    @property
    def config(self):
        if not hasattr(self, '_config') or self._config is None:
            self.build_config()
        return self._config

    @property
    def pretty_config(self):
        return C.to_yaml(self.config)

    @classmethod
    def _new_config(cls)->C:
        return C.create()

    @classmethod
    def default_config(cls) -> C:
        return C.merge(
            super().default_config(),
            cls._new_config()
        )

    def build_config(self) -> C:
        # build config from self._init_kwargs
        self._config = C.merge(self.default_config(), self._init_kwargs)


    """ build and handle modules """
    def build_modules(self, *args):
        self.main = None

    def parameters(self, recurse: bool = True):
        assert self._initialized
        return super().parameters(recurse)

    def to(self, *args, **kwargs):
        assert self._initialized
        return super().to(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        assert self._initialized
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        assert self._initialized
        return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)


    """ utilities """
    def __str__(self) -> str:
        return torch.nn.Module.__str__(self)

    def __hash__(self) -> int:
        return hash(f'THISISANOPWITHID:{self._id}')

    def __copy__(self):
        # TODO: shallow copy a node will only copy the config
        raise NotImplementedError

    def __deepcopy__(self):
        # TODO: deep copy a node will copy the config and the modules
        raise NotImplementedError("deepcopy is not supported for now")

    def new(cls, name=None):
        op = super().__new__(cls)
        op.__init__()
        if name is not None:
            op._name = name
        return  op

    def __new__(cls, *args, name=None, **kwargs): # Calling the operator will generates a new line of code ..
        return cls.NODE_MAP(cls.new(cls, name), *args, **kwargs)


    def reuse(self, *args, **kwargs):
        return self.NODE_MAP(self, *args, **kwargs)

    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError("Please use reuse() to reuse an operator")