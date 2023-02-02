# the basic class of building modules from configs based on the input types
from .operator import Operator, nodes_to_types
from .unification import unify, TypeInferenceFailure
from .basetypes import TupleType
from omegaconf import OmegaConf as C
        

def match_types(input_node, type_lists):
    for idx, types in enumerate(type_lists):
        try:
            unify(input_node, types, input_node)[-1]
        except TypeInferenceFailure:
            continue
        return idx
    return -1


class Factory(Operator):
    KEYS = []
    OPERATORS = []
    NAMES = []

    def build_modules(self, *input_types):
        #return super().build_modules(*args)
        if len(input_types) == 1:
            input_types = input_types[0]

        idx = match_types(input_types, self.KEYS)

        if idx == -1:
            raise TypeError(f"no matching operator for {self.inp_types} in {self.__class__.__name__}\n Factory: {self.KEYS}")
        self.main = self.OPERATORS[idx](*self.default_inp_nodes, **self._init_kwargs)

    @classmethod
    def _new_config(cls):
        outs = {}
        for k, v, name in zip(cls.KEYS, cls.OPERATORS, cls.NAMES):
            outs[name] = v.default_config()
        return C.create(outs)

    def build_config(self):
        return super().build_config()

    @classmethod
    def register(cls, module, type, name=None):
        cls.KEYS.append(type)
        cls.OPERATORS.append(module)

        name = name or module.__name__
        assert name not in cls.NAMES, f"module name {name} already exists for Factory {cls.__name__}"
        cls.NAMES.append(name)
        
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _type_inference(self, *inp_types):
        #return self.OPERATORS._type_inference(*inp_types)
        return self.main.get_output().get_type()

def test():
    class Encoder(Factory):
        pass

    from .types.tensor import MLP, Tensor1D, TensorType
    from .types.image import ConvNet, ImageType

    Encoder.register(MLP, ImageType, 'mlp')
    Encoder.register(ConvNet, Tensor1D, 'conv')

    inp = TensorType(4, 3, 32, 32, data_dims=3)
    print(C.to_yaml(Encoder.default_config()))
    encoder = Encoder(inp)
    print(encoder)

        
if __name__ == '__main__':
    test()