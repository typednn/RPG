import termcolor
from ..basetypes import AttrType
from omegaconf import OmegaConf as C
from .tensor import TensorType, Type, Arrow
from ..unification import unify, TypeInferenceFailure

from ..operator import Operator
from .. import utils


class Pointcloud(TensorType):
    PREFIX = 'Pointcloud'
    def __init__(self, *size, data_dims=2, dtype=None, device=None):
        super().__init__(*size, data_dims=data_dims, dtype=dtype, device=device)

class PointDict(AttrType):
    xyz: Pointcloud('...', 3, 'N')
    rgb: Pointcloud('...', 'D', 'N')
    agent: TensorType('...', 'M')

    
from nn.modules.point import PointNet as OldPointNet
class PointNet(Operator):
    arrow = Arrow(PointDict(), TensorType('...', 'dim')) #TODO: if possible, infer it from the config

    @classmethod
    def _new_config(cls):
        return utils.yaml2omega(OldPointNet.dc)

    def build_modules(self, inp_type: "PointDict"):
        import gym
        assert inp_type.xyz.size[1] == 3
        inp_dict = {
            'xyz': gym.spaces.Box(-1, 1., shape=(3,)),
            'rgb': gym.spaces.Box(-1, 1., shape=(inp_type.rgb.size[-2],)),
            'agent': gym.spaces.Box(-1., 1., shape=(inp_type.agent.size[-1],))
        }
        cfg = utils.omega2yaml(self.config)
        self.main = OldPointNet(inp_dict, cfg=cfg).to(inp_type.xyz.device)
            
    def _type_inference(self, inp_types) -> Type:
        #return super()._type_inference(*inp_types)
        return inp_types.agent.new(*inp_types.agent.batch_shape(), *self.main._output_shape)


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
    x = PointDict(Pointcloud(256, 3, 'N'), Pointcloud(256, 1, 'N'), TensorType(256, 'N'))
    y = PointDict(xyz=Pointcloud(256, 3, 'N'), rgb=Pointcloud(256, 1, 'N'), agent=TensorType(256, 'N'))
    assert str(x) == str(y)

    try:
        x = PointDict(xyz=Pointcloud(257, 3, 'N'), agb=Pointcloud(257, 4, 'N'), agent=TensorType(256, 'N'))
    except TypeInferenceFailure as e:
        print('correctly get error', termcolor.colored(e.__class__.__name__ + ' ' + str(e), 'red'))

    Z = AttrType(xyz=Pointcloud('...', 'D', 'N'))
    Z.check_compatibility(x)

#def test_attrnode():
#    x = 
def test_pointnet():
    from ..functors import Tuple
    inp = PointDict(Pointcloud('...', 3, 'N'), Pointcloud('...', 4, 'N'), TensorType('...', 10))

    t = Tuple(inp, inp)
    xyz, _ = t
    xyz = xyz.xyz
    print(xyz)
    exit(0)

    pointnet = PointNet(inp)
    print(pointnet)
    exit(0)


if __name__ == '__main__':
    test_pointnet()
    #test_pointcloud()
    #test_pointdict()