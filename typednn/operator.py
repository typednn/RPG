# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import inspect
import copy
import typing
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


def get_left_value(frame):
    frame = frame[0]
    return inspect.getframeinfo(frame).code_context[0].strip().split("=")[0].strip()


OPID = 0

class Operator(OptBase):
    INFER_SHAPE_BY_FORWARD=False
    arrow = Arrow(VariableArgs("...", None), Type("output")) # TYPE annotation of the forward funciton
    #N_OUTPUT=None

    def __init__(self, *args, name=None, _trace_history=None, **kwargs) -> None:
        super().__init__()
        # TODO: store the lineno for debugging
        self._init_keys, self._init_args, self._init_kwargs = self.process_args_kwargs(*args, **kwargs)

        self._name = name or self.__class__.__name__
        self._default_inp_nodes = None
        #self._default_inp_nodes = [Node.from_val(i) for i in self._init_args]
        self.clear()

        global OPID
        self._id = OPID
        OPID += 1

        from .utils import exception_with_traceback
        self.call_frame = self.find_caller()
        self.left_value = get_left_value(self.call_frame)
        self._trace_history = ('\n' + _trace_history() if _trace_history is not None else '') + '\n\ninit at\n' + exception_with_traceback(self.call_frame[0]) 

    def process_args_kwargs(self, *args, **kwargs):
        config = self.default_config()
        config_args = {}

        args = list(args)
        keys = [f'args{i}' for i in range(len(args))]

        for k, v in kwargs.items():
            if k in config:
                config_args[k] = v
            else:
                args.append(v)
                keys.append(k)
        return keys, args, C.create(config_args)

    def match_input(self, *args, **kwargs):
        inps = list(args)
        for i in range(len(args), len(self._init_args)):
            if self._init_keys[i] is None:
                raise ValueError(f"missing input {i}")
            if self._init_keys[i] in kwargs:
                inps.append(kwargs.pop(self._init_keys[i]))
            else:
                raise ValueError(f"missing input {self._init_keys[i]}")
        if len(kwargs) > 0:
            raise ValueError(f"extra inputs {kwargs.keys()}")
        return inps, kwargs

    @property
    def default_inp_nodes(self):
        if self._default_inp_nodes is None:
            self._default_inp_nodes = [Node.from_val(i) for i in self._init_args]
        return self._default_inp_nodes

    def find_caller(self, key='OPERATORS'):
        # find the caller of this function
        for frame in inspect.stack():
            # hack ..
            if (self.__class__.__name__ in frame[4][0] or key in frame[4][0]):
                return frame
        raise ValueError("cannot find the caller of this function")

    def get_trace(self):
        return self._trace_history

    def clear(self):
        self._lazy_init = False
        self._config = None
        self._default_out = None

    def init(self):
        if not self._lazy_init:
            self._lazy_init = True
            self.build_config() # configure it

            try:
                self.main
                has_main = True
            except AttributeError:
                has_main = False
            inp_types = nodes_to_types(self.default_inp_nodes)
            if not has_main:
                self.build_modules(*inp_types)

    def myassert(self, cond, msg='', errorType=ValueError):
        frame_assert(cond, msg, self.get_trace, errorType)

    # get the output node of the operator based on input nodes ..
    def shadow(self, *input_nodes: typing.List[Node], default=False, **kwargs) -> Node:
        # TODO: for replicate operators.
        # self.myassert(default, "left value inference is not implemneted")
        input_nodes, _ = self.match_input(*input_nodes, **kwargs)
        input_nodes = [Node.from_val(i) for i in input_nodes]
        if default:
            name = self.left_value
        else:
            for frame in inspect.stack():
                if ('.shadow' in frame[4][0]):
                    break
            name = get_left_value(frame)

        if ',' in name:
            name = '[' + name + ']'
        return CallNode(self.get_output_type_by_input(*input_nodes), self, name=name, input_nodes=input_nodes)

    def _type_inference(self, *input_types) -> Type:
        from .unification import TypeInferenceFailure
        error = None
        try:
            _, _, _out_type = self.arrow.unify(*input_types)
        except TypeInferenceFailure as e:
            error = e
        frame_assert(
            error is None,
            f"cannot infer the output type of {self._name} with input {input_types}.\n{error}", self.get_trace, error.__class__)
        return _out_type

    def _type_inference_after_init(self, *input_types) -> Type:
        if self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            return self._get_type_from_output(output, *input_types)
        return self._type_inference(*input_types)

    def _get_type_from_output(self, output, *args):
        inp = args[0]
        out_shape = inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return out_shape

    # required for infering the output type
    def get_output_type_by_input(self, *input_nodes, force_init=False):
        if force_init:
            self.init()
        if self._lazy_init: #if initliaized, use another way to do the type inference ..
            input_types = nodes_to_types(input_nodes)
            return self._type_inference_after_init(*input_types)
        else:
            input_types = [i._meta_type for i in input_nodes]
            return self._type_inference(*input_types)

    # wrappers
    def get_output(self) -> Node: # out type when input are feed ..
        # TODO: if possible, extract the default name from the line calling the method
        if self._default_out is None:
            self._default_out = self.shadow(*self.default_inp_nodes, default=True)
        return self._default_out

    def __iter__(self):
        return iter(self.get_output())

    def __getattr__(self, name: str):
        # assert name in self._modules, f"module {name} is not found in {self.__class__}"
        error = None
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            error = e

        if error:
            if name != 'main':
                error_again = None
                try:
                    attr = getattr(self.get_output(), name)
                except AttributeError as e:
                    error_again = e
                if error_again is not None:
                    raise error
            else:
                raise error
        return attr


    """ calling, this is not for the graph construct """
    def __call__(self, *args, **kwargs):
        self.init()
        args, kwargs = self.match_input(*args, **kwargs)

        #TODO: this only checks for the default input nodes ..
        for a, b in zip(self.default_inp_nodes, args):
            a = a.get_type()
            
            if isinstance(a, Type) and a.instance(b) is None:
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                from .utils import tensor2error
                frame_assert(False, f"input {tensor2error(b)} does not match the required input type {a} for {info}", self.get_trace, TypeError)
                
        out = super().__call__(*args, **kwargs)
        # TODO: check output type
        out_type = self.get_output().get_type()
        assert out_type.instance(out) is not None, f"output {out} does not match the required output type {out_type}"
        return out

        
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

    def forward(self, *args, **kwargs):
        assert self.main is not None, "please either override this function or set the module of the class"
        try:
            return self.main(*args, **kwargs)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            # print(self.call_frame)
            
            frame_assert(False,
                         f"error in forward function of {self.__class__}:\n{e} with\n {str(tb)}",
                         self.get_trace,
                         e.__class__)
            

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


    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        if self._lazy_init:
            import logging
            logging.warning(f"reconfiguring a module {self._name} that has already been initialized, this is not recommended")

        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self.clear()
        

    def compile(self, *args, **kwargs):
        return self.get_output().compile(*args, **kwargs)
        

    """ utilities """
    def __str__(self) -> str:
        #out = super().__str__()
        self.init()
        out = torch.nn.Module.__str__(self)
        return out + f" -> {self.get_output().get_type()}"

    def __hash__(self) -> int:
        return hash(f'THISISANOPWITHID:{self._id}')

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self):
        raise NotImplementedError("deepcopy is not supported for now")