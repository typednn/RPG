import torch

from .node import Node, InputNode, CallNode #, ArrowNode
from .basetypes import Arrow
from .operator import Code
from omegaconf import OmegaConf as C


# Notice that here Function itself is not an operator  
# Calling it will generate an operator
# funcnode will generate a FuncNode

class Function(Code):
    nodes = None
    named_input = None
    operators = None

    @classmethod
    def new(
        cls,
        context,
        inputs=None, # a named input order
        name=None
    ) -> None:
        self = object.__new__(cls)

        input_nodes = context['inputs']

        self.nodes = context['nodes']
        self.named_input = {}
        if inputs is None:
            for idx, k in enumerate(input_nodes):
                self.named_input[f'[{idx}]'] = k
        else:
            assert len(inputs) == len(input_nodes)
            self.named_input = {name: node for node, name in inputs.items()}

        self.arrow = Arrow(
            **{k:v._meta_type for k, v in self.named_input.items()},
            out=self.output_node._meta_type
        )

        self.operators = {name: module for module, name in context['submodules'].items()}

        assert len(self.operators) == len(context['submodules'])
        Code.__init__(self)
        if name is not None:
            self._name = name
        return self

    def clone(self):
        raise NotImplementedError

    @property
    def output_node(self):
        if not hasattr(self, '_output_node'):
            self._output_node = list(self.nodes.values())[-1]
        return self._output_node

    def __call__(self, *args, key=None, **kwargs):
        from .abstraction import CallNode
        return CallNode(self, *args, key=key or self._name, **kwargs)

    def as_node(self):
        return self(*self.named_input.values(), key='as_node')

    def build_modules(self, *input_types):
        #self.output_node.get_type() # this will directly get the output ..
        self.type_inference(*input_types, init=True)
        self.main = torch.nn.ModuleDict(self.operators)

    def type_inference(self, *input_types, init=False):
        context = {
            node: type for node, type in zip(self.named_input.values(), input_types)
        }
        if not init:
            context['_do_not_init'] = True
        return self.output_node.get_type(context)

    def reconfig(self, **kwargs):
        #return super().reconfig(**kwargs)
        outs = {}
        for k, v in kwargs.items():
            if k in self.operators:
                self.operators[k].reconfig(**v)
            else:
                outs[k] = v
        super().reconfig(**outs)

    def build_config(self):
        config = C.merge(self.default_config(), self._init_kwargs)
        for name, module in self.operators.items():
            config[name] = module.config
            config[name]['_type'] = module.__class__.__name__
        self._config = C.create(config)

    def forward(self, *inps):
        context = {}
        if len(inps) != len(self.named_input):
            raise ValueError(f'Expected {len(self.named_input)} inputs, got {len(inps)}')
        for node, b in zip(self.named_input.values(), inps):
            context[node] = b
        return self.output_node.evaluate(context)

    def __str__(self) -> str:
        #TODO: output nodes; (with input and output types)
        self.init()
        out = ''
        for idx, (name, k) in enumerate(self.operators.items()):
            out += f' ({idx}).[TODO:leftvalue?] of {name}: ' + str(k).replace('\n', '\n   ') + '\n'

        out = out + 'Inputs: ' + ', '.join([str(self.named_input[k]) for k in self.named_input])
        for lineno, i in enumerate(self.nodes):
            line = f'\n{lineno}. {i._name} @ {i._id} <- '
            line += i.print_line()
            line += ' ' * max(80 -len(line), 0) + ' # ' + str(i.get_type())

            out += line
        #out = 'Input: ' + ', '.join(map(str, self.inp_types)) + '\n'
        return out

    # def __deepcopy__(self):
    #     raise NotImplementedError("deepcopy is not supported for ModuleGraph")


def abstract(
    node: Node,
    context=None,
    config=None,
    build=True,
    inputs=None,
    **kwargs
) -> Function:
    """
    search backward to collect all operators and computation nodes
    """

    if build:
        if config is None:
            config = {}
        if context is None:
            context = {}

        for key in ['opname_count', 'inputs', 'nodes', 'visited', 'submodules']:
            if key not in context:
                context[key] = {}
    else:
        assert context is not None, "context should be provided for non-root nodes"

    if node in context['visited']:
        assert context['visited'][node], "cyclic dependency detected"
        return context
    context['visited'][node] = False


    if inputs is None or node not in inputs:
        parents = node.get_parents()
        for i in parents:
            abstract(i, build=False, context=context, inputs=inputs, config=config)

    if isinstance(node, InputNode) or (inputs is not None and node in inputs):
        context['inputs'][node] = node

    if isinstance(node, Node):
        context['nodes'][node] = node

        if isinstance(node, CallNode):
            # when op is a module
            code = node.code
            if code not in context['submodules']:
                # remove duplicated name
                name = code._name
                #if name in context['opname_count']:

                val_count = context['opname_count'].get(name, 0) + 1
                context['opname_count'][name] = val_count
                if val_count > 1:
                    name = name+ '_' + str(val_count)

                code.reconfig(**config.get(name, {}))
                context['submodules'][code] = name

    context['visited'][node] = True
    if not build:
        return context
    else:
        return Function.new(context, inputs=inputs, **kwargs)