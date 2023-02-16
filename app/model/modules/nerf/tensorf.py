# data class
# camera -> ray -> xyz points
# field: xyz -> functions

# ray, rgb, sigma + camera -> xyz points 
import torch
from typednn.operator import ArrowNode, Arrow
from typednn.functors import asmodule, torchop
from .types import Field, camera_class, ray_class


# class Render(ArrowNode):
#     arrow =  Arrow(Field(), ) # given a field
#     #def __init__(self, ):
    
    
def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


@torchop(distance_scale=25, rayMarch_weight_thres=0.0001, is_train=True, white_bg=True)
def render(
    ray_dir: ray_class.dir,

    xyz_sampled,
    z_vals,
    ray_valid,

    sigma_field,
    feature_field,
    render_module,

    config=None
):
    dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
    sigma = torch.zeros(xyz_sampled.shape[:2], device=xyz_sampled.device)
    rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

    #if ray_valid.any():
    # xyz_sampled = self.normalize_coord(xyz_sampled) # TODO: do this inside the sample ..
    # sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
    # validsigma = self.feature2density(sigma_feature)
    # sigma[ray_valid] = validsigma
    if ray_valid.any():
        #TODO: remember to add back the feature2density
        sigma[ray_valid] = sigma_field(xyz_sampled[ray_valid])

    alpha, weight, bg_weight = raw2alpha(sigma, dists * config.distance_scale)
    app_mask = weight > config.rayMarch_weight_thres

    if app_mask.any():
        #app_features = self.compute_appfeature(xyz_sampled[app_mask])
        app_features = feature_field(xyz_sampled[app_mask])
        valid_rgbs = render_module(xyz_sampled[app_mask], ray_dir[app_mask], app_features)
        rgb[app_mask] = valid_rgbs

    acc_map = torch.sum(weight, -1)
    rgb_map = torch.sum(weight[..., None] * rgb, -2)

    if config.white_bg or (config.is_train and torch.rand((1,))<0.5):
        rgb_map = rgb_map + (1. - acc_map[..., None])

    
    rgb_map = rgb_map.clamp(0,1)

    with torch.no_grad():
        depth_map = torch.sum(weight * z_vals, -1)
        depth_map = depth_map + (1. - acc_map) * ray_dir[..., -1]

    return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

@asmodule
def render(field: Field(), ray: ray_class, render_module):
    pass