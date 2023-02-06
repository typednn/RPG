import typing
import termcolor
import inspect
from ..node import InputNode, NodeBase
from ..types import TupleType, AttrType
from .utils import Tuple, Dict


# TODO: add support to quickly wrap a function into a node 

def moduledef(func: typing.Mapping):
    annotation = func.__annotations__
    #print(annotation)
    input_nodes = {}
    for k, v in annotation.items():
        input_nodes[k] = InputNode(v, name=k)

    output = func(*input_nodes.values())
    if isinstance(output, tuple):
        output = Tuple(*output)
    if isinstance(output, dict) and not isinstance(output, NodeBase):
        output = Dict(**output)
    return output.compile(input_order=input_nodes)


def test_module_define():
    import torch
    from ..types import TensorType
    from ..types.image import ConvNet
    from .utils import FlattenBatch

    image = torch.zeros(1, 5, 224, 224, device='cuda:0')
    #pass

    image_type = TensorType('B', 5, 224, 224, data_dims=3)
    @moduledef
    def mymodule(inp1: image_type, inp2: image_type):
        img = ConvNet(FlattenBatch(inp1))
        img2 = img.shadow(FlattenBatch(inp2))
        return {'img': img, 'img2': img2}
    print(mymodule)

    try:
        mymodule(image)
    except ValueError as e:
        print(termcolor.colored('Error Expected!', 'green'), termcolor.colored(str(e), 'red'))

    x2 = mymodule(image, image)
    assert torch.allclose(x2.img, x2.img2)


if __name__ == '__main__':
    test_module_define()