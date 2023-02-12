# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2

"""
The operator is the container of the neural networks 

- arrow: the type of the operator
- get_output_type_by_input(self, *args, **kwargs) -> type: type inference of the operator  

- __call__(self, *args, **kwargs) -> data

- reconfig(**kwargs): compute the 
- init(*args, **kwargs) -> None

"""
import torch
from torch import nn
from omegaconf import OmegaConf as C
from torch.nn import Module
from .basetypes import Arrow, Type, VariableArgs
from .node import Node, CallNode, nodes_to_types
from .utils import frame_assert
from .unification import unify
from typing import Mapping, Any


class OptBase(Module):
    @classmethod
    def default_config(cls) -> C:
        return C.create()

OPID = 0

class Operator(OptBase):
    INFER_SHAPE_BY_FORWARD=False
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton
    #N_OUTPUT=None

    def __init__(self, name=None) -> None: 
        super().__init__()
        self._name = name or self.__class__.__name__

        global OPID
        self._id = OPID
        OPID += 1

        self._init_kwargs = C.create()
        self.clear()

    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        if self._initialized:
            #import logging
            #logging.warning(f"reconfiguring a module {self._name} that has already been initialized, this is not recommended")
            raise NotImplementedError("Can't config an initalized operator.")
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self.clear()

    def clear(self):
        # self._default_inp_nodes = None
        # self._default_out = None
        self._initialized = False
        self._config = None


    def init(self, *inp_nodes):
        if not self._initialized:
            self.type_inference(*inp_nodes) # infer the meta types 
            self._initialized = True
            try:
                self.main
                has_main = True
            except AttributeError:
                has_main = False
            inp_types = nodes_to_types(inp_nodes)
            if not has_main:
                self.build_modules(*inp_types)


    def _type_inference(self, *input_types) -> Type:
        from .unification import TypeInferenceFailure

        if self._initialized and self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            inp = input_types[0]
            return inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])

        error = None
        try:
            _, _, _out_type = self.arrow.unify(*input_types)
        except TypeInferenceFailure as e:
            error = e
        self.myassert(error is None, f"cannot infer the output type of {self._name} with input {input_types}.\n{error}", error.__class__)
        return _out_type

    # required for infering the output type
    def type_inference(self, *input_nodes):
        if self._initialized: #if initliaized, use another way to do the type inference ..
            input_types = nodes_to_types(input_nodes)
        else:
            input_types = [i._meta_type for i in input_nodes]
        return self._type_inference(*input_types)

    def myassert(self, cond, msg='', errorType=ValueError):
        #frame_assert(cond, msg, self.get_trace, errorType)
        frame_assert(cond, msg, lambda: '', errorType)


    def forward(self, *args, **kwargs):
        assert self.main is not None, "please either override this function or set the module of the class"
        try:
            return self.main(*args, **kwargs)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.myassert(False,  f"error in forward function of {self.__class__}:\n{e} with\n {str(tb)}", e.__class__)
            


    # @property
    # def default_inp_nodes(self):
    #     if self._default_inp_nodes is None:
    #         from .node import ShadowNode
    #         self._default_inp_nodes = [Node.from_val(i) for i in self._init_args]
    #     return self._default_inp_nodes


    # def get_trace(self):
    #     return self._trace_history

    # # get the output node of the operator based on input nodes ..
    # def shadow(self, *input_nodes: typing.List[Node], default=False, **kwargs) -> Node:
    #     # TODO: for replicate operators.
    #     # self.myassert(default, "left value inference is not implemneted")
    #     input_nodes, _ = self.match_input(*input_nodes, **kwargs)
    #     input_nodes = [Node.from_val(i) for i in input_nodes]
    #     if default:
    #         name = self.left_value
    #     else:
    #         for frame in inspect.stack():
    #             if ('.shadow' in frame[4][0]):
    #                 break
    #         name = get_left_value(frame)

    #     if ',' in name:
    #         name = '[' + name + ']'
    #     return CallNode(self.get_output_type_by_input(*input_nodes), self, name=name, input_nodes=input_nodes)

    # # wrappers
    # def get_output(self) -> Node: # out type when input are feed ..
    #     # TODO: if possible, extract the default name from the line calling the method
    #     if self._default_out is None:
    #         self._default_out = self.shadow(*self.default_inp_nodes, default=True)
    #     return self._default_out

    # def __iter__(self):
    #     return iter(self.get_output())

    # def __getattr__(self, name: str):
    #     # assert name in self._modules, f"module {name} is not found in {self.__class__}"
    #     error = None
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError as e:
    #         error = e

    #     if error:
    #         if name != 'main':
    #             error_again = None
    #             try:
    #                 attr = getattr(self.get_output(), name)
    #             except AttributeError as e:
    #                 error_again = e
    #             if error_again is not None:
    #                 raise error
    #         else:
    #             raise error
    #     return attr

    # """ calling, this is not for the graph construct """
    # def __call__(self, *args, **kwargs):
    #     self.init()
    #     args, kwargs = self.match_input(*args, **kwargs)

    #     #TODO: this only checks for the default input nodes ..
    #     for a, b in zip(self.default_inp_nodes, args):
    #         a = a.get_type()
            
    #         if isinstance(a, Type) and a.instance(b) is None:
    #             info = '\n' + str(self)
    #             info = info.replace('\n', '\n' + '>' * 10)
    #             from .utils import tensor2error
    #             frame_assert(False, f"input {tensor2error(b)} does not match the required input type {a} for {info}", self.get_trace, TypeError)
                
    #     out = super().__call__(*args, **kwargs)
    #     # TODO: check output type
    #     out_type = self.get_output().get_type()
    #     assert out_type.instance(out) is not None, f"output {out} does not match the required output type {out_type}"
    #     return out

        
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
        self.init()
        return super().parameters(recurse)

    def to(self, *args, **kwargs):
        self.init()
        return super().to(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.init()
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.init()
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

    def __new__(cls, *args, **kwargs):
        op = super().__new__(cls)
        op.__init__()
        return CallNode(op, *args, **kwargs)