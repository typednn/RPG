from torch import nn
import torch
from ..operator import Operator
from .tensor import TensorType, VariableArgs, Type
from ..functors import Flatten, Seq, Linear, FlattenBatch


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
    #from nn.scg import *
    #inp = TensorType('N', 3, 224, 224, data_dims=3)
    #inp = TensorType('...', 3, 224, 224, data_dims=3)
    inp = TensorType('N', 5,224,224, data_dims=3)
    #inp = ImageType
    flattenb = FlattenBatch(inp)
    conv = ConvNet(flattenb, layer=4)
    flatten = Flatten(conv)
    linear = Linear(flatten, dim=20)
    #print(seq)
    # print(seq)
    #print(flatten.collect_modules())
    
    # print(linear.out)
    graph = linear.collect_modules()
    seq = Seq(flattenb, conv, flatten, linear)

    image = inp.sample()
    print(graph(image).shape)
    print(seq(image).shape)
    assert torch.allclose(graph(image), seq(image))


if __name__ == '__main__':
    test_conv()