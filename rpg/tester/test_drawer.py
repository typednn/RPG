# Let's put the utility here
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_o3d_camera(int, image_res):
    import open3d as o3d
    fx, fy = int[0, 0], int[1, 1]
    cx, cy = int[0, 2], int[1, 2]
    w, h = image_res[1], image_res[0]
    cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return cam

def get_int(image_res):
    fov = 0.23
    image_res = image_res
    int = np.array([
        - np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
        - np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
        [0, 0, 1]
    ])
    return np.linalg.inv(int)

def draw_geometries(objects, mode='human', img_res=(512, 512), int=None, ext=None):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, visible=False)
    if isinstance(objects, tuple) or isinstance(objects, list):
        for geom in objects:
            vis.add_geometry(geom)
    else:
        vis.add_geometry(objects)
    ctr = vis.get_view_control()

    if int is None:
        int = get_int(img_res)

    cam_param = get_o3d_camera(int, img_res)
    o3d_cam = o3d.camera.PinholeCameraParameters()
    o3d_cam.intrinsic = cam_param
    if ext is None:
        from tools.utils import lookat
        R, t = lookat([0.5, 0.5, 0.5], 0., 0., 3.)
        ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
        ext = np.linalg.pinv(ext)
    o3d_cam.extrinsic = ext #self.get_ext()

    ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    image = np.uint8(np.asarray(image) * 255)
    return image

        
def np2pcd(xyz):
    import open3d as o3d
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

img = draw_geometries([np2pcd(np.random.rand(100, 3))], mode='rgb_array')
plt.imshow(img)
plt.savefig('test.png')