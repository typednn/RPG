# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2

import typing
import torch
from torch import nn
from omegaconf import OmegaConf as C
from torch.nn import Module
from .basetypes import Arrow, Type
from .node import Node, nodes_to_types


class OptBase(Module):
    def default_config(self) -> C:
        return C.create()

class Operator(OptBase):
    INFER_SHAPE_BY_FORWARD=False
    arrow: typing.Optional[Arrow] = None # TYPE annotation of the forward funciton
    N_OUTPUT=None

    def __init__(self, *args, name=None, **kwargs) -> None:
        super().__init__()
        # TODO: store the lineno for debugging

        self._lazy_init = False
        self._lazy_config = False
        self._init_args, self._init_kwargs = args, C.create(kwargs)
        self._name = name or self.__class__.__name__
        self._default_out = None


    def init(self):
        if not self._lazy_init:
            self._lazy_init = True
            args = self._init_args
            self.build_config() # configure it

            self.default_inp_nodes = [Node.from_val(i) for i in args]
            if self.main is None:
                self.build_modules(*nodes_to_types(self.default_inp_nodes))
            self.execute(*self.default_inp_nodes)

    # get the output node of the operator based on input nodes ..
    def execute(self, *input_nodes: typing.List[Node]) -> Node:
        # construct a shadow node
        input_types = nodes_to_types(input_nodes)
        return Node(parent=self, n_childs=self.get_n_output(*input_nodes), inp_types=input_types)

    """ type inference """
    def _infer_by_arrow(self, *inp_types):
        _, _, self._out_type = self.arrow.unify(inp_types)
        return self._out_type

    # required for infering the number of output 
    def get_n_output(self, *input_nodes):
        if self.arrow is not None:
            return len(self.arrow.out)
        if self.N_OUTPUT is not None:
            return self.N_OUTPUT
        return 1

    def _type_inference(self, *inp_types):
        raise NotImplementedError

    # required for infering the output type
    def get_output_type_by_input(self, *inp_types):
        assert not hasattr(self, '_out_type'), "type_inference can only be called once"

        if self.INFER_SHAPE_BY_FORWARD:
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in inp_types]
            output = self.forward(*shapes)
            _out_type = self._get_type_from_output(output, *inp_types)
        elif self.arrow is not None:
            _out_type = self._infer_by_arrow(*inp_types)
        else:
            assert hasattr(self, '_type_inference'), f"please either override the type_inference function or set the arrow of the class {self.__class__}"
            _out_type = self._type_inference(*inp_types)
        return _out_type

    # wrappers
    @property
    def out(self) -> Node: # out type when input are feed ..
        #return Node(self._out_type, self, None)
        if self._default_out is None:
            self._default_out = self.execute(*self.default_inp_nodes)
        return self._default_out

    def __iter__(self):
        return iter(self.out)


    """ calling, this is not for the graph construct """
    def __call__(self, *args, **kwargs):
        self.init()
        for a, b in zip(self.inp_types, args):
            if isinstance(a, Type) and not a.instance(b):
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                raise TypeError(f"input type {a} does not match the input {b} for {info}")
        out = super().__call__(*args, **kwargs)
        # TODO: check output type
        return out


        
    """ config system """
    @property
    def config(self):
        if not hasattr(self, '_config'):
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
            super().default_config(cls),
            cls._new_config()
        )

    def forward(self, *args, **kwargs):
        assert self.main is not None, "please either override this function or set the module of the class"
        return self.main(*args, **kwargs)

    def build_config(self) -> C:
        if not hasattr(self, '_config'):
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


    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))


    def configure(self, build=True, context=None, config=None):
        """
        search backward to collect all modules
        """
        if hasattr(self, '_configured'):
            _all_modules = getattr(self, '_configured')
            if _all_modules is None:
                raise Exception("configure() is called twice for one module; there are probably some circular dependencies in the module tree.")
            return _all_modules
        if not hasattr(self, '_configured'):
            setattr(self, '_configured', None)

        if build:
            context = {'_name_count': {}, '_inps':  []}

        inps = context['_inps']
        for i in self._init_args:
            if isinstance(i, Operator):
                # TODO: configure
                i.configure(build=False, context=context)
            else:
                if hasattr(i, '_trace'):
                    i._trace['module'].configure(False, context=context)
                else:
                    if i not in inps:
                        inps.append(i)

        val_count = context['_name_count'][self._name] = context['_name_count'].get(self._name, 0) + 1
        name = self._name
        if val_count > 1:
            name = self._name + '_' + str(val_count)
        context[name] = self
        self._name = name # rename the module to avoid name conflict
        
                    
        if not build:
            setattr(self, '_configured', inps)
            return inps
        else:
            context.pop('_name_count')
            return ModuleGraph(context, context.pop('_inps'))
    

    """ path """
    def __str__(self) -> str:
        #out = super().__str__()
        self.init()
        out = torch.nn.Module.__str__(self)
        return out + f"\nOutputType: {self.out_type}"

    def _get_type_from_output(self, output, *args):
        inp = self.inp_types[0]
        out_shape = inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return out_shape



class ModuleGraph(Operator):
    def init(self):
        if not self._lazy_init:
            self._lazy_init = True
            args = self._init_args
            self.models = args[0]
            self.inp_types = args[1]
            self.build_config()
            for k, v in self.models.items():
                v.init()


    def forward(self, *inps):
        assert len(self.inp_types) == 1, "only support one input for now"
        context = {}
        for a, b in zip(self.inp_types, inps):
            assert a.instance(b)
            context[a] = b

        for module_name, module in self.models.items():
            inps = []
            for k in module._init_args:
                if isinstance(k, Operator):
                    val = context[k._name]
                elif isinstance(k, Type):
                    if hasattr(k, '_trace'):
                        trace = k._trace
                        val = context[trace['module']._name]
                        if 'index' in trace and trace['index'] is not None:
                            val = val[trace['index']]
                    else:
                        val = context[k]
                else:
                    val = k
                inps.append(val)
            out = context[module_name] = module(*inps)
        return out

    def __str__(self) -> str:
        self.init()
        out = 'Input: ' + ', '.join(map(str, self.inp_types)) + '\n'
        for k, v in self.models.items():
            out += f' ({k}): ' + str(v).replace('\n', '\n   ') + '\n'
        return out

    def get_context(self):
        return self._init_args[0]

    def _type_inference(self, *args):
        context = self.get_context()
        for k, v in context.items():
            pass
        return v.out

    def build_config(self):
        config = dict(
        )
        self._config = config

        context = self.get_context()
        for idx, module in context.items():
            config[idx] = module.config
            config[idx]['_type'] = module.__class__.__name__
        self._config = C.create(config)
        
        
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