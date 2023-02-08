from typednn import Factory
from typednn.types.tensor import MLP, Tensor1D, TensorType
from typednn.types.image import ConvNet, ImageType


class Encoder(Factory):
    @classmethod
    def _common_config(cls):
        return {
            'out_dim': 64,
        }

Encoder.register(MLP, Tensor1D, 'mlp')
Encoder.register(ConvNet, ImageType, 'conv')

"""
Image:
- ResNeXt

ImplicitFunc + BBOX + CameraMatrix + PixelMask -> Camera:
- Volume-based
- Embedding-based

Trajectory: 
- GPT2
- RNN

Pointcloud:
- PointNet 

MultiviewImage:
- for NERF


Multihead-Deocder (change dimension and then split)
"""
