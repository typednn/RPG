from torch import nn

import torch
from ..operator import Operator
from .tensor import TensorType, VariableArgs, Type
from ..functors import Flatten, Seq, Linear, FlattenBatch, Tuple



# TD = Type('D')
# TN = Type('N')
# TM = Type('M')
ImageType = TensorType('...', 'D', 'N', 'M', data_dims=3)


class ConvNet(Operator):
    INFER_SHAPE_BY_FORWARD=True

    @classmethod
    def _new_config(cls):
        return dict(
            layer=3,
            hidden=512,
            out_dim=32,
        )

    def build_modules(self, inp_type):
        try:
            int(inp_type.data_shape().total())
        except TypeError:
            raise TypeError(f'ConvNet requires a fixed input shape but receives {str(inp_type)}')
        assert inp_type.data_dims is 3
        C = inp_type.channel_dim

        num_channels = self.config.hidden
        self.main = nn.Sequential(
            nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, self.config.out_dim, 3, stride=2), nn.ReLU()
        ).to(inp_type.device)


def test_conv():
    inp = TensorType('N', 5,224,224, data_dims=3)

    flattenb = FlattenBatch(inp)
    conv = ConvNet(flattenb, layer=4)
    flatten = Flatten(conv)
    
    linear = Linear(flatten, dim=20)
    linear3, _ = Tuple(linear, flatten)
    linear2 = Linear(linear3, dim=10, name='output')

    graph = Tuple(linear, linear2).configure()
    

    seq = Seq(flattenb, conv, flatten, linear, linear2)
    image = inp.sample()
    from omegaconf import OmegaConf as C
    assert torch.allclose(graph(image)[1], seq(image))

    print("OK!")
    


if __name__ == '__main__':
    test_conv()