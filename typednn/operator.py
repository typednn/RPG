# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import torch
from torch import nn
from torch.nn import Module
from .basetypes import Arrow, Type, VariableArgs
from .node import Node, CallNode
from omegaconf import OmegaConf as C
from .context import Context
from .utils import frame_assert
from .unification import unify
from typing import Mapping, Any, Callable


class ConfigurableBase:
    @classmethod
    def default_config(cls) -> C:
        return C.create()

class ArrowNode(Node, ConfigurableBase):
    arrow = None

    def __init__(self, *parents, **kwargs) -> None:
        self.parents = list(parents)
        meta_type = self._get_meta_type(*parents)
        self._init_kwargs = C.create()
        Node.__init__(self, meta_type, **kwargs)
        # self.contexts = set() # record the context that this node is called

    def _get_meta_type(self, *parents):
        return self.arrow

    def _type_inference(self, *input_types, context) -> Type:
        raise NotImplementedError(f"type inference is not supported for {self.__class__.__name__}")

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

    def _get_config(self) -> C:
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

class Code(ArrowNode, Module):
    """
    Modify pytorch module into an arrow node
    The module will be initialized once we want to infer the type of the node for the first time
    """


    INFER_SHAPE_BY_FORWARD=False
    NODE_MAP=CallNode
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton

    def __new__(cls, *args, name=None, **kwargs): # Calling the operator will generates a new line of code ..
        default_config = cls.default_config()
        op: "Code" = cls.new(cls, name)
        inputs = {}
        init_config = {}
        for k, v in kwargs.items():
            if k in default_config:
                init_config[k] = v
            else:
                inputs[k] = v
        op.reconfig(**init_config)
        return cls.NODE_MAP(op, *args, **inputs)

    def reuse(self, *args, **kwargs):
        return self.NODE_MAP(self, *args, **kwargs)

    def __init__(self) -> None: 
        Module.__init__(self)
        ArrowNode.__init__(self)
        self._name = self.__class__.__name__

        self._config = None
        self._module = None

    def _get_evaluate(self, *parents_callable, context):
        assert len(parents_callable) is 0
        self._context = context
        return self.forward

    def _type_inference(self, *input_types, context: Context) -> Type:
        from .unification import TypeInferenceFailure
        if context is not None and self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self._get_module(context)(*shapes)
            inp = input_types[0]
            return inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return super()._type_inference(*input_types, context=context)

    """ build and handle modules """
    def _get_module(self, context: Context):
        if self._module is None:
            self._module = self.build_model(*[context.type[i] for i in self._input_nodes])
        return self._module

    def build_model(self, *args):
        return torch.nn.Identity()

    def forward(self, *args, **kwargs):
        return self._get_module(self._context)(*args, **kwargs)

    def new(cls, name=None):
        op = super().__new__(cls)
        op.__init__()
        if name is not None:
            op._name = name
        return  op

    """ utilities where self.context is used """
    def __str__(self) -> str:
        return torch.nn.Module.__str__(self)

    def reconfig(self, **kwargs):
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self._config = None
            
    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
        return self._config

    @property
    def pretty_config(self):
        return C.to_yaml(self.config)