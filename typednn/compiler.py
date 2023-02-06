#TODO: support detach node
#TODO: provide detail level for print and visualization

from .operator import Operator
from omegaconf import OmegaConf as C
from .node import Node, InputNode, CallNode


class ModuleGraph(Operator):
    def __init__(
        self,
        *args,
        name=None,
        input_order=None,
        **kwargs
    ) -> None:
        super().__init__(*args, name=name, **kwargs)

        args = self._init_args
        self.context = args[0]
        self.input_nodes = self.context['inputs']
        self.submodules = self.context['submodules']
        self.nodes = self.context['nodes']
        self.output_node = list(self.nodes.keys())[-1]

        
        self.named_input = {}

        if input_order is None:
            input_order = self.input_nodes
            for idx, k in enumerate(self.input_nodes):
                self.named_input[f'[{idx}]'] = k
        else:
            self.named_input = input_order
        self.default_inp_nodes = self.named_input.values()
        
    def find_caller(self):
        import inspect
        for frame in inspect.stack():
            if 'compile' in frame[4][0] and 'self' not in frame[4][0]:
                return frame
        raise ValueError("cannot find the caller of this function")

    def init(self):
        if not self._lazy_init:
            self._lazy_init = True
            self.build_config()
            for i in self.submodules:
                i.init()
            self.inp_types = [i.get_type() for i in self.default_inp_nodes]
            self.build_modules()

    def build_modules(self):
        import torch
        self.main =  torch.nn.ModuleDict({name: k for k, name in self.submodules.items()})

    def forward(self, *inps, **kwargs):
        context = {}
        for (k, node), b in zip(self.named_input.items(), inps):
            context[node] = b
        for k, v in kwargs.items():
            if k not in self.named_input:
                raise ValueError(f'input {k} is not defined')
            node = self.named_input[k]
            if node in context:
                raise ValueError(f'input {k} is already defined')
            context[node] = v

        if len(context) != len(self.named_input):
            raise ValueError(f'input length is not correct, expected {list(self.named_input.keys())}, but only got {list(context.keys())}')

        return self.output_node.evaluate(context)

    def __str__(self) -> str:
        #TODO: output nodes; (with input and output types)
        self.init()
        out = ''
        for idx, (k, name) in enumerate(self.submodules.items()):
            out += f' ({idx}).{k.left_value}  of {name}: ' + str(k).replace('\n', '\n   ') + '\n'

        out = out + 'Inputs: ' + ', '.join([str(self.input_nodes[k]) for k in self.default_inp_nodes])
        for lineno, i in enumerate(self.nodes):
            line = f'\n{lineno}. {i._name} @ {i._id} <- '
            line += i.print_line()
            line += ' ' * max(80 -len(line), 0) + ' # ' + str(i.get_type())

            out += line
        #out = 'Input: ' + ', '.join(map(str, self.inp_types)) + '\n'
        return out

    def get_context(self):
        return self._init_args[0]

    def _type_inference(self, *args):
        return self.output_node.get_type()

    def build_config(self):
        config = dict(
        )
        self._config = config

        for module, name in self.submodules.items():
            config[name] = module.config
            config[name]['_type'] = module.__class__.__name__
        self._config = C.create(config)

    
    def __deepcopy__(self):
        raise NotImplementedError("deepcopy is not supported for ModuleGraph")


def compile(node: Node, context=None, config=None, build=True, **kwargs) -> ModuleGraph:
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


    parents = node.get_parents()
    for i in parents:
        compile(i, build=False, context=context, config=config)

    if isinstance(node, InputNode):
        context['inputs'][node] = node
    if isinstance(node, Node):
        context['nodes'][node] = node

        if isinstance(node, CallNode):
            # when op is a module
            op = node.module
            if op not in context['submodules']:
                # remove duplicated name
                name = op._name
                op.reconfig(**config.get(name, {}))
                if name in context['opname_count']:
                    val_count = context['opname_count'].get(name, 0) + 1
                    context['opname_count'][name] = val_count
                    if val_count > 1:
                        name = name+ '_' + str(val_count)
                context['submodules'][op] = name

    context['visited'][node] = True
    if not build:
        return context
    else:
        return ModuleGraph(context, **kwargs)