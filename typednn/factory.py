# the basic class of building modules from configs based on the input types
import termcolor
from .operator import Operator, nodes_to_types
from .unification import unify, TypeInferenceFailure
from omegaconf import OmegaConf as C
        

def match_types(input_node, type_lists):
    for idx, types in enumerate(type_lists):
        try:
            unify(input_node, types, input_node)[-1]
        except TypeInferenceFailure:
            continue

        print(termcolor.colored(str(input_node) + ' matches ' + str(types) + ' of index ' + str(idx), 'green'))
        return idx
    return -1


class Factory(Operator):
    KEYS = []
    OPERATORS = []
    NAMES = []

    def __init__(self, *args, name=None, _trace_history=None, **kwargs) -> None:
        super().__init__(*args, name=name, _trace_history=_trace_history, **kwargs)
        self._selected_module = None

    def get_selected_module(self):
        if self._selected_module is None:
            self.select_module(*nodes_to_types(self.default_inp_nodes))
        return self._selected_module
    
    def get_sub_config(self, idx):
        import copy
        config = copy.deepcopy(self.config)
        module_config = config.get(self.NAMES[idx])
        for i in self.NAMES:
            config.pop(i)
        return C.merge(config, module_config)

    def select_module(self, *input_types):
        if len(input_types) == 1:
            input_types = input_types[0]
        self.module_idx = match_types(input_types, self.KEYS)
        idx = self.module_idx

        if idx == -1:
            raise TypeError(
                f"no matching operator for {input_types} in"
                f" {self.__class__.__name__}\n Factory: {self.KEYS}"
            )
        get_trace = lambda: f'Factory of {self.OPERATORS[idx]} for input type {input_types} built at' + self.get_trace()
        self._selected_module = self.OPERATORS[idx](
            *self.default_inp_nodes, _trace_history=get_trace, 
            **self.get_sub_config(idx),
        )

    def build_modules(self, *args):
        print('inited here ..')
        module = self.get_selected_module()
        module.init()
        self.main = module

    def reconfig(self, **kwargs):
        super().reconfig(**kwargs)
        assert not self._lazy_init
        self.get_selected_module().reconfig(
            **self.get_sub_config(self.module_idx))

    @classmethod
    def _common_config(cls):
        return {}

    @classmethod
    def _new_config(cls):
        common = cls._common_config()

        outs = {}
        for v, name in zip(cls.OPERATORS, cls.NAMES):
            outs[name] = v.default_config()
            for k in common:
                if k in outs[name]:
                    outs[name].pop(k)
        outs.update(common)
        return C.create(outs)

    @classmethod
    def register(cls, module, type, name=None):
        cls.KEYS.append(type)
        cls.OPERATORS.append(module)

        name = name or module.__name__
        assert name not in cls.NAMES, f"module name {name} already exists for Factory {cls.__name__}"
        cls.NAMES.append(name)

    def _type_inference(self, *input_types):
        return self.get_selected_module()._type_inference(*input_types)

    def __str__(self) -> str:
        self.init()
        main_str = str(self.main).replace('\n', '\n  ')
        return f'Factory of {self.__class__.__name__} for input type {self.input_types}  (\n  {main_str}\n)'


def test():
    class Encoder(Factory):
        @classmethod
        def _common_config(cls):
            return {
                'out_dim': 64,
            }

    from .types.tensor import MLP, Tensor1D, TensorType
    from .types.image import ConvNet, ImageType
    from .functors import Linear, Flatten

    Encoder.register(MLP, Tensor1D, 'mlp')
    Encoder.register(ConvNet, ImageType, 'conv')

    try:
        inp = TensorType(4, 3, 32, 32, data_dims=3)
        encoder = Encoder(inp)
        print(encoder)
    except RuntimeError as e:
        print('failed')
        print('error message', termcolor.colored(e, 'red'))
    print('ok')


    inp = TensorType(4, 3, 150, 150, data_dims=3)
    print(C.to_yaml(Encoder.default_config()))
    encoder = Encoder(inp)
    flatten = Flatten(encoder)
    linear = Linear(flatten, dim=32)
    graph = linear.compile(config=dict(Encoder=dict(out_dim=100, conv=dict(layer=6))))
    print(graph.pretty_config)
    
    #print(encoder.get_type())
    assert str(encoder.get_type()) == "Tensor3D(4 : 100,1,1)"
    assert str(graph.get_output().get_type()) == "Tensor(4 : 32)"
    
        
if __name__ == '__main__':
    test()