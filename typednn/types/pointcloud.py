import termcolor
from ..basetypes import AttrType
from .tensor import TensorType, Type
from ..unification import unify, TypeInferenceFailure


class Pointcloud(TensorType):
    PREFIX = 'Pointcloud'
    def __init__(self, *size, data_dims=2, dtype=None, device=None):
        super().__init__(*size, data_dims=data_dims, dtype=dtype, device=device)

class PointDict(AttrType):
    xyz: Pointcloud('...', 'D', 'N')
    agent: TensorType('...', 'N')

#a = PointcloudType()

def test_pointcloud():
    a = Pointcloud('...', 3, 'N')

    out = unify(a, TensorType(100, 3, 5, data_dims=2), a, update_name=False)[2]
    assert str(out) == "Pointcloud(100 : 3,5)", f"got {out}"

    try:
        unify(a, TensorType(100, 3, 5, data_dims=1), a, update_name=False)[2]
    except TypeInferenceFailure as e:
        pass

    b = unify(a, TensorType('...', 3, 'M', data_dims=2), a, update_name=True, queryA=True)[1]
    assert str(b) == "Tensor2D((?)... : 3,N)", f"got {b}"


def test_pointdict():
    x = PointDict(Pointcloud(256, 3, 'N'), TensorType(256, 'N'))
    y = PointDict(xyz=Pointcloud(256, 3, 'N'), agent=TensorType(256, 'N'))
    assert str(x) == str(y)

    try:
        x = PointDict(xyz=Pointcloud(257, 3, 'N'), agent=TensorType(256, 'N'))
    except TypeInferenceFailure as e:
        print('correctly get error', termcolor.colored(e.__class__.__name__ + ' ' + str(e), 'red'))


if __name__ == '__main__':
    #test_pointcloud()
    test_pointdict()