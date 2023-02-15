from torch import nn
import termcolor

import torch
from ..context import Context
from ..operator import Code
from .tensor import TensorType, VariableArgs, Type
from ..node import Node
from ..functors import Flatten, Linear, FlattenBatch, Tuple

ImageType = TensorType('...', 'D', 'N', 'M', data_dims=3)


class ConvNet(Code):
    INFER_SHAPE_BY_FORWARD=True

    @classmethod
    def _new_config(cls):
        return dict(
            layer=4,
            hidden=512,
            out_dim=32,
        )

    def _type_inference(self, input_types, context) -> Type:
        if context is None:
            assert isinstance(input_types, TensorType)
            return TensorType(*input_types.batch_shape(), self.config.out_dim, 'N', 'M', data_dims=3)
        else:
            return super()._type_inference(input_types, context=context)

    def build_model(self, inp_type: "ImageType"):
        try:
            int(inp_type.data_shape().total())
        except TypeError:
            raise TypeError(f'ConvNet requires a fixed input shape but receives {str(inp_type)}')
        assert inp_type.data_dims is 3
        C = inp_type.channel_dim

        config = self.config
        num_channels = config.hidden
        return nn.Sequential(
            nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
            *sum([[nn.Conv2d(num_channels, num_channels, 3, 2), nn.ReLU()] for _ in range(max(config.layer-4, 0))], []),
            nn.Conv2d(num_channels, config.out_dim, 3, stride=2), nn.ReLU()
        ).to(inp_type.device)


def test_conv():
    inp = TensorType('...', 5, 224, 224, data_dims=3)
    image = torch.zeros(5, 224, 224, device='cuda:0')
    assert inp.instance(image)

    # assert inp.instance(torch.zeros([5, 5, 224, 224]))
    # assert inp.instance(torch.zeros([5, 224, 224]))

    flattenb = FlattenBatch(inp)
    
    conv = ConvNet(flattenb, layer=4)
    flatten = Flatten(conv)
    
    linear = Linear(flatten, dim=20)

    linear3, other = Tuple(linear, flatten)
    linear2 = Linear(linear3, dim=10)

    out = Tuple(linear, linear2)


    graph = out.compile(config=dict(Linear=dict(dim=36)))

    #print(graph._type_inference(TensorType(32, 5, 224, 224, data_dims=3), context=Context()))
    #exit(0)

    #seq = Seq(flattenb, conv, flatten, linear, linear2)

    image = inp.sample()
    from omegaconf import OmegaConf as C
    
    #print(image.shape)
    #image = torch.zeros(5, 224, 224, device='cuda:0')
    seq = [flattenb, conv, flatten, linear, linear2]
    x = image
    for i in seq:
        x = i.eval(x)

    assert torch.allclose(graph.forward(image)[1], x), f"{image.shape}"

    img = torch.tensor([1., 2., 3.])

    try:
        graph.forward(img)
    except TypeError as e:
        print(termcolor.colored(str(e), 'red'))
    print("OK!")

    # graph.init()
    graph._get_module(graph.default_context)

    print('conv parameters', len(list(conv.op.parameters())))
    print('graph parameters', len(list(graph.parameters())))
    


if __name__ == '__main__':
    test_conv()