# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import torch
from torch import nn
from torch.nn import Module
from .configurable import Configurable
from .basetypes import Arrow, Type, VariableArgs
from .node import Node, CallNode
from omegaconf import OmegaConf as C
from .utils import frame_assert
from .unification import unify
from typing import Mapping, Any, Callable


OPID = 0


class ArrowNode(Node, Module, Configurable):
    """
    Major components:
    - infer the meta type
    - config and meta config
    - init
    - do type inference
    """
    arrow = None

    def __init__(
        self,
        *parents, # sub module
        name=None,
        trace=None,
        **kwargs
    ) -> None:
        Module.__init__(self)
        self.parents = list(parents)
        meta_type = self._get_meta_type(*parents)
        Node.__init__(meta_type, name, trace, **kwargs)

    def _get_meta_type(self, *args):
        return self.arrow

    # Node interface 
    def get_parents(self):
        return self.parents

    def _get_type(self, context):
        return super()._get_type(context)

    def print_line(self):
        return self._name

    def _evaluate(self, context) -> Callable:
        forwards = [i.evaluate(context) for i in self.parents]
        return self._get_callable(*forwards)

    def type_inference(self, *input_types) -> Type:
        return self._type_inference(*input_types)


    # specific for the arrow node ..
    def _get_callable(self, *parents_callable):
        raise NotImplementedError

    def _type_inference(*input_types) -> Type:
        raise NotImplementedError




class Code(ArrowNode):
    INFER_SHAPE_BY_FORWARD=False
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton
    NODE_MAP=CallNode

    def __new__(cls, *args, name=None, **kwargs): # Calling the operator will generates a new line of code ..
        return cls.NODE_MAP(cls.new(cls, name), *args, **kwargs)

    def reuse(self, *args, **kwargs):
        return self.NODE_MAP(self, *args, **kwargs)

    def __init__(self) -> None: 
        super().__init__()
        self._name = self.__class__.__name__
        self._initialized = False

    # hooks
    def set_input_nodes(self, *input_nodes, keys=None):
        self._input_nodes = input_nodes
        self._input_keys = keys

    def clone(self, shallow=True):
        raise NotImplementedError

    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        if self._initialized:
            #import logging
            #logging.warning(f"reconfiguring a module {self._name} that has already been initialized, this is not recommended")
            raise NotImplementedError("Can't config an initalized operator.")
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self.clear()

    def init(self, *inp_types):
        if not self._initialized:
            self._initialized = True
            self.build_modules(*inp_types)

    def _type_inference(self, *input_types) -> Type:
        from .unification import TypeInferenceFailure
        if self._initialized and self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            inp = input_types[0]
            return inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        _, _, _out_type = self.arrow.unify(*input_types)
        return _out_type

    #def forward(self, *args, **kwargs):
    #    return self.main(*args, **kwargs)

    """ build and handle modules """
    def build_modules(self, *args):
        self.main = None

    """ utilities """
    def __str__(self) -> str:
        return torch.nn.Module.__str__(self)

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
    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError("Please use reuse() to reuse an operator")


    

    # pytorch 
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