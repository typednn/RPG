import torch
from typednn.types import AttrType, TensorType, Type, TupleType
from typednn.functors import asfunc, torchmethod
from .nerf_utils import get_ray_directions
from typednn import Class

# Training data, scene
class Ray(Class):
    origin: TensorType('B:ray',  3)
    dir: TensorType('B:ray', 3)
    index: TensorType('B:ray', 2, dtype=torch.long) # return [scene, camera] index of the field
    pixels: TensorType('B:ray', 2)

    @torchmethod(is_train=True, n_samples=-1, step_ratio=2.0, step_size=None)
    def sample(
        self, 
        gridSize: TensorType('B:ray', 3, device=torch.long), 
        aabb: TensorType('B:ray', 2, 3), 
        near_far: TensorType('B:ray', 2), 
        config=None
    ):
        # copied from tensorf sample_ray
        rays_o = self.origin
        rays_d = self.dir

        B = len(rays_o)
        n_samples = config.n_samples


        aabbSize = aabb[..., 1] - aabb[..., 0]
        invaabbSize = 2.0 / aabbSize
        units=aabbSize / (gridSize-1)
        stepsize = torch.mean(units, axis=-1) * config.step_ratio

        if n_samples < 0:
            # used for testing ..
            n_samples = (torch.linalg.norm(aabbSize, dim=-1) / stepsize).long() + 1
        else:
            n_samples = torch.ones(B, device=rays_o.device, dtype=torch.long) * n_samples

        #near, far = config.near_far
        near, far = near_far[:, 0], near_far[:, 1]
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (aabb[..., 1] - rays_o) / vec
        rate_b = (aabb[..., 0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        # we want to start from t_min compute N_samples .. 
        max_samples = n_samples.max() # let's padding 

        rng = torch.arange(max_samples)[None].float()
        if config.is_train:
            rng = rng.repeat(B, 1)
            rng += torch.rand_like(rng[:,[0]])

        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((aabb[..., 0]>rays_pts) | (rays_pts>aabb[..., 1])).any(dim=-1)


        return rays_pts, interpx, ~mask_outbbox


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

    @torchmethod()
    def ray(self, camid: TensorType('B:ray', 1), pixels: TensorType('B:ray', 2)) -> ray_class:
        index = torch.cat([self.imageid[camid[..., 0]], camid], dim=-1)

        camid = camid[:, 0]
        inv_intrinsic = self.inv_intrinsic[camid]
        c2w = self.c2w[camid]
        dir = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=-1)
        dir = inv_intrinsic @ dir[..., None].double()
        dir = dir / torch.norm(dir, dim=-2, keepdim=True)
        dir = c2w[..., :3, :3].double() @ dir

        return ray_class.new(c2w[..., :3, 3], dir[..., 0].float(), index, pixels)

    def new(self, c2w, intrinsic, imageid, inv_intrinsic=None):
        if inv_intrinsic is None:
            inv_intrinsic = torch.inverse(intrinsic.double())
        return super().new(c2w, intrinsic, imageid, inv_intrinsic)
         

camera_class = Camera()


def test_camera():
    from .nerf_utils import load_single_image_dataset
    data = load_single_image_dataset('lego')
    for i in data:
        print(i, data[i].shape)
        data[i] = data[i].cuda()

    c2w = data['poses'][..., :3, :]
    intrinsic = data['intrinsic'][None,:].expand(len(c2w), -1, -1)
    all_ray = data['rays'].reshape(len(c2w), -1, 6)

    iid = torch.zeros(len(c2w), 1, dtype=torch.long).cuda()
    camera = camera_class.new(c2w, intrinsic, iid)
    #print(camera.shape, camera._base_type)

    #node = camera_class.ray()
    print(data['intrinsic'].shape)
    w, h =  data['intrinsic'][:2, 2] * 2
    print(w, h)

    from kornia import create_meshgrid
    # try the first camera only
    grid = create_meshgrid(int(h), int(w), normalized_coordinates=False)[0].cuda() + 0.5
    cam_id = torch.zeros(*grid.shape[:2], 1, dtype=torch.long).cuda()

    grid = grid.reshape(-1, 2)
    cam_id = cam_id.reshape(-1, 1)

    node = camera_class.ray(TensorType('B:cam', 1), ray_class.pixels)
    out_ray = node.eval(camera, cam_id, grid)

    assert torch.allclose(out_ray.origin, all_ray[0][:, :3])
    assert torch.allclose(out_ray.dir, all_ray[0][:, 3:], rtol=1e-3, atol=1e-7)
    print(all_ray[0][:, 3:])

    # test the whole ray
    grid = grid[None, :].expand(20, -1, -1)
    cam_id = cam_id[None, :].expand(20, -1, -1)
    cam_id = cam_id + torch.arange(20, dtype=torch.long)[:, None, None].cuda()

    grid = grid.reshape(-1, 2)
    cam_id = cam_id.reshape(-1, 1)

    out_ray = node.eval(camera, cam_id, grid)
    print(out_ray.origin.shape)
    print(out_ray.dir.shape)
    print(grid.shape)
    print(all_ray[:20, :, :3].shape)
    N = 20

    assert torch.allclose(out_ray.origin, all_ray[:N, :, :3].reshape(-1, 3))
    assert torch.allclose(out_ray.dir, all_ray[:N, :, 3:].reshape(-1, 3), rtol=1e-3, atol=1e-7)

    # now test ray

    

if __name__ == '__main__':
    # python3 -m app.model.modules.nerf.types.camera
    test_camera()