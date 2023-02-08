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
    print(output._meta_type)
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

    assert str(mymodule.arrow) == 'inp1:Tensor3D(B : 5,224,224)->inp2:Tensor3D(B : 5,224,224)->out:AttrType(img=Tensor3D(B : 32,N,M), img2=Tensor3D(B : 32,N,M))'

    try:
        mymodule(image)
    except ValueError as e:
        print(termcolor.colored('Error Expected!', 'green'), termcolor.colored(str(e), 'red'))

    x2 = mymodule(image, image)
    assert torch.allclose(x2.img, x2.img2)

    print('mymodule output type', mymodule.get_output().print_line())
    #assert str(mymodule.get_output().get_type()) == str(AttrType(img=image_type, img2=image_type))
    print(mymodule.get_type())
    assert str(mymodule.get_output().get_type()) == str(AttrType(img=TensorType('B', 32, 12, 12, data_dims=3), img2=TensorType('B', 32, 12, 12, data_dims=3)))
    print(mymodule.arrow)


if __name__ == '__main__':
    test_module_define()