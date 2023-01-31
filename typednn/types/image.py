from torch import nn
from ..operator import Operator
from .tensor import TensorType, VariableArgs, Type


# TD = Type('D')
# TN = Type('N')
# TM = Type('M')
ImageType = TensorType('...', 'D', 'N', 'M', data_dims=3)


class ConvNet(Operator):
    @classmethod
    def _new_config(cls):
        return dict(
            layer=3,
            hidden=512,
            out_dim=32,
        )

    def build_modules(self, inp_type):
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
    inp = TensorType('...', 3, 224, 224, data_dims=3)
    #inp = ImageType
    print(inp.size)
    conv = ConvNet(inp, layer=4)
    print(conv)


if __name__ == '__main__':
    test_conv()