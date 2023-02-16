import torch
from typednn.types import AttrType, TensorType, Type
from typednn.functors import asfunc, asmethod, torchmethod
from .nerf_utils import get_ray_directions
from typednn import Class

# Training data, scene

class Ray(Class):
    origin: TensorType('...',  3)
    dir: TensorType('...',  3)
    index: TensorType('...',  1, dtype=torch.long) # return which camera comes from
    pixels: TensorType('...',  2)

    # @torchmethod
    # def sample():
    #     pass

    # @torchmethod
    # def integration(self, ) -> TensorType('...', 3):
    #     pass


class Camera(Class):
     """
     c2w is the pose in TensorF implementation
     intrinsic: must be [[focal, 0, w/], [0, focal, h/], [0, 0, 1]]
     """
     c2w: TensorType('...cam', 3, 4)
     intrinsic: TensorType('...cam', 3, 3)
     imageid: TensorType('...cam', 1, dtype=torch.long)

     @torchmethod
     def ray(self, camid: TensorType('...', 1), pixels: TensorType('...', 2)) -> Ray():
         intrinsic = torch.gather(self.intrinsic, 0, camid.unsqueeze(-1)) 
         c2w = torch.gather(self.c2w, 0, camid.unsqueeze(-1))
         dir = intrinsic @ pixels
         dir = c2w @ (dir / torch.norm(dir, dim=-1, keepdim=True))
         return Ray.new(self.c2w[..., :3, 3], dir, camid, pixels)


def test_ray():
    ray = Ray(origin=TensorType(10, 3))


def test_camera():
    from .nerf_utils import load_single_image_dataset
    data = load_single_image_dataset('lego')
    for i in data:
        print(i, data[i].shape)

    Camera.new(
        data['c2w'],
        data['intrinsic'][None,:].repeat(),
        data['imageid']
    )

if __name__ == '__main__':
    # python3 -m app.model.modules.nerf.types.camera
    test_ray()
    test_camera()