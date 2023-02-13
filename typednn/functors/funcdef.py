import typing
import termcolor
import inspect
from ..node import InputNode, NodeBase
from ..types import TupleType, AttrType
from .utils import Tuple, Dict
from ..abstraction import asfunc

    

def test_module_define():
    import torch
    from ..types import TensorType
    from ..types.image import ConvNet
    from .utils import FlattenBatch

    image = torch.zeros(1, 5, 224, 224, device='cuda:0')
    #pass

    image_type = TensorType('B', 5, 224, 224, data_dims=3)

    @asfunc
    def mymodule(inp1: image_type, inp2: image_type):
        img = ConvNet(FlattenBatch(inp1))
        img2 = img.reuse(FlattenBatch(inp2))
        return {'img': img, 'img2': img2}

    assert str(mymodule.arrow) == 'inp1:Tensor3D(B : 5,224,224)->inp2:Tensor3D(B : 5,224,224)->out:AttrType(img=Tensor3D(B : 32,N,M), img2=Tensor3D(B : 32,N,M))', str(mymodule.arrow)

    try:
        mymodule.forward(image)
    except ValueError as e:
        print(termcolor.colored('Error Expected!', 'green'), termcolor.colored(str(e), 'red'))

    x2 = mymodule.forward(image, image)
    assert torch.allclose(x2.img, x2.img2)

    #node = mymodule(image_type, image_type)
    node = mymodule.as_node()
    print('mymodule output type', node.print_line())
    #assert str(mymodule.get_output().get_type()) == str(AttrType(img=image_type, img2=image_type))
    print(node.get_type())
    assert str(node.get_type()) == str(AttrType(img=TensorType('B', 32, 12, 12, data_dims=3), img2=TensorType('B', 32, 12, 12, data_dims=3)))


def test_module_define2():
    from ..types import TensorType, MLP
    tensortype = TensorType('B', 'M', data_dims=1)

    """
    Ok in the end we just need one things:
    - given an operator, we can define its computation graph with one type of input nodes 
    - but we can change or modify the input nodes when really initalize the operator

    - ok, the simplest way to implement this is the detach node; and we modify the detach node so that we can change the input node .. I feel that I am the genious..
    - support revise init_args for reconfig.. that's all
    """

    @asfunc
    def mymodule(inp1: tensortype):
        return MLP(inp1)
    
    #print(mymodule)
    print(mymodule.as_node()._meta_type)
    #print(mymodule.pretty_config)
    #exit(0)
    mymodule.reconfig(MLP=dict(out_dim=512))

    tensortype2 = TensorType('B', 100, data_dims=1)
    node2 = mymodule(tensortype2)

    #print(mymodule(tensortype2))
    #print(node2._meta_type)
    print(node2)

if __name__ == '__main__':
    # test_module_define()
    test_module_define2()