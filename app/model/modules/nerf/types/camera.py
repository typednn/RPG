import torch
from typednn.types import AttrType, TensorType, Type
from typednn.functors import asfunc, torchmethod
from .nerf_utils import get_ray_directions
from typednn import Class

# Training data, scene

class Ray(Class):
    origin: TensorType('B:ray',  3)
    dir: TensorType('B:ray', 3)
    index: TensorType('B:ray', 1, dtype=torch.long) # return which camera comes from
    pixels: TensorType('B:ray',  2)


ray_class = Ray()

class Camera(Class):
    """
    c2w is the pose in TensorF implementation
    intrinsic: must be [[focal, 0, w/], [0, focal, h/], [0, 0, 1]]
    """
    c2w: TensorType('B:cam', 3, 4, data_dims=2)
    intrinsic: TensorType('B:cam', 3, 3, data_dims=2)
    imageid: TensorType('B:cam', 1, dtype=torch.long)
    inv_intrinsic: TensorType('B:cam', 3, 3, data_dims=2)

    @torchmethod
    def ray(self, camid: TensorType('B:ray', 1), pixels: TensorType('B:ray', 2)) -> ray_class:
        camid = camid[:, 0]
        inv_intrinsic = self.inv_intrinsic[camid]
        c2w = self.c2w[camid]
        dir = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=-1)
        dir = inv_intrinsic @ dir[..., None].double()
        dir = dir / torch.norm(dir, dim=-2, keepdim=True)
        dir = c2w[..., :3, :3].double() @ dir
        return ray_class.new(c2w[..., :3, 3], dir[..., 0].float(), camid[..., None], pixels)

    def new(self, c2w, intrinsic, imageid, inv_intrinsic=None):
        if inv_intrinsic is None:
            inv_intrinsic = torch.inverse(intrinsic.double())
        return super().new(c2w, intrinsic, imageid, inv_intrinsic)
         

camera_class = Camera()


def test_ray():
    ray = Ray(origin=TensorType(10, 3))


def test_camera():
    from .nerf_utils import load_single_image_dataset
    data = load_single_image_dataset('lego')
    for i in data:
        print(i, data[i].shape)

    c2w = data['poses'][..., :3, :]
    intrinsic = data['intrinsic'][None,:].expand(len(c2w), -1, -1)
    all_ray = data['rays'].reshape(len(c2w), -1, 6)

    iid = torch.zeros(len(c2w), 1, dtype=torch.long)
    camera = camera_class.new(c2w, intrinsic, iid)
    #print(camera.shape, camera._base_type)

    #node = camera_class.ray()
    print(data['intrinsic'].shape)
    w, h =  data['intrinsic'][:2, 2] * 2
    print(w, h)

    from kornia import create_meshgrid
    # try the first camera only
    grid = create_meshgrid(int(h), int(w), normalized_coordinates=False)[0] + 0.5
    cam_id = torch.zeros(*grid.shape[:2], 1, dtype=torch.long)

    grid = grid.reshape(-1, 2)
    cam_id = cam_id.reshape(-1, 1)
    #out_ray = camera_class.ray.method.func(camera, cam_id, grid)


    node = camera_class.ray(ray_class.index, ray_class.pixels)
    out_ray = node.eval(camera, cam_id, grid)

    assert torch.allclose(out_ray.origin, all_ray[0][:, :3])
    assert torch.allclose(out_ray.dir, all_ray[0][:, 3:], rtol=1e-3, atol=1e-7)
    print(all_ray[0][:, 3:])

    

if __name__ == '__main__':
    # python3 -m app.model.modules.nerf.types.camera
    test_ray()
    test_camera()