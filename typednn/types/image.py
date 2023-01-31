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
        C = inp_type.size[-3]
        try:
            C = int(C)
        except TypeError as e:
            raise TypeError(str(e) + f"\n    The actual channel is Type {C}")

        num_channels = self.config.hidden
        self.main = nn.Sequential(
            nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, self.config.out_dim, 3, stride=2), nn.ReLU()
        ).to(inp_type.device)

    def _get_type_from_output(self, output, *args):
        inp = self.inp_types[0]
        out_shape = inp.new(*inp.batch_shape(), *output.shape[-3:])
        return out_shape


def test_conv():
    #from nn.scg import *
    inp = TensorType('...', 3, 224, 224, data_dims=3)
    #inp = ImageType
    print(inp.size)
    conv = ConvNet(inp, layer=4)
    print(conv)


if __name__ == '__main__':
    test_conv()