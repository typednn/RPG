# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import torch
from torch import nn
from torch.nn import Module
from .configurable import ConfigurableBase
from .basetypes import Arrow, Type, VariableArgs
from .node import Node, CallNode
from omegaconf import OmegaConf as C
from .utils import frame_assert
from .unification import unify
from typing import Mapping, Any, Callable

OPID = 0

class ArrowNode(Node, Module, ConfigurableBase):
    arrow = None

    def __init__(self, *parents, **kwargs) -> None:
        Module.__init__(self)
        self.parents = list(parents)
        meta_type = self._get_meta_type(*parents)
        self._init_kwargs = C.create()
        Node.__init__(self, meta_type, **kwargs)

    def _get_meta_type(self, *parents):
        return self.arrow

    def _type_inference(self, *input_types, context) -> Type:
        return context[self].unify(*input_types)[-1]

    # Node interface 
    def get_parents(self):
        return self.parents

    def print_line(self):
        return self._name

    # NODE interface: will initalize the operators
    def _get_type(self, *args, context):
        return self.arrow

    def _get_evaluate(self, *parents_callable, context=None):
        raise NotImplementedError

    def __copy__(self):
        # TODO: shallow copy a node will only copy the config
        raise NotImplementedError

    def __deepcopy__(self):
        # TODO: deep copy a node will copy the config and the modules
        raise NotImplementedError("deepcopy is not supported for now")


    def _get_config(self, *args, context) -> C:
        # build config from self._init_kwargs
        return C.merge(self.default_config(), self._init_kwargs)


    @classmethod
    def _new_config(cls)->C:
        return C.create()

    @classmethod
    def default_config(cls) -> C:
        return C.merge(
            super().default_config(),
            cls._new_config()
        )


class Code(ArrowNode):
    INFER_SHAPE_BY_FORWARD=False
    NODE_MAP=CallNode
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton

    def __new__(cls, *args, name=None, **kwargs): # Calling the operator will generates a new line of code ..
        return cls.NODE_MAP(cls.new(cls, name), *args, **kwargs)

    def reuse(self, *args, **kwargs):
        return self.NODE_MAP(self, *args, **kwargs)

    def __init__(self) -> None: 
        super().__init__()
        self._name = self.__class__.__name__

    def _get_evaluate(self, *parents_callable, context):
        assert len(parents_callable) is 0
        return context.module[self]

    def _get_type(self, context):
        return self.arrow

    def _type_inference(self, *input_types, context) -> Type:
        from .unification import TypeInferenceFailure
        if self._initialized and self.INFER_SHAPE_BY_FORWARD:
            input_types = input_types
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            inp = input_types[0]
            return inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return super()._type_inference(*input_types, context=context)

    """ build and handle modules """
    def _get_module(self, **input_types):
        return None

    """ utilities """
    def __str__(self) -> str:
        return torch.nn.Module.__str__(self)

    def new(cls, name=None):
        op = super().__new__(cls)
        op.__init__()
        if name is not None:
            op._name = name
        return  op

    # """ code for manage computation graph """
    def reconfig(self, context=None, **kwargs):
        #TODO: rewrite this part
        if context is None:
            context = self.context

        assert self not in context.module.dict
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        if self in context.config.dict:
            del context.config.dict[self]

    def config(self, context):
        context = context or self.context
        return context.config[self]

    @property
    def pretty_config(self):
        return C.to_yaml(self.config)

    # # pytorch 
    # def parameters(self, recurse: bool = True):
    #     assert self._initialized
    #     return super().parameters(recurse)

    # def to(self, *args, **kwargs):
    #     assert self._initialized
    #     return super().to(*args, **kwargs)

    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     assert self._initialized
    #     return super().load_state_dict(state_dict=state_dict, strict=strict)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     assert self._initialized
    #     return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)