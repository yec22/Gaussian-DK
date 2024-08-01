#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera,  pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, render_hdr = False, render_lightup = False, depth_threshold = None, iteration = None, spherify = False, target_camera = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold,
        iteration=iteration,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            hdr_rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # [N, 3]
            light_feature = pc.get_lightness_features # [N, LC]
            colors_precomp = torch.cat([hdr_rgb, light_feature], dim=-1) # [N, 3+LC]
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image_conbine, radii, pixels = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image_hdr = rendered_image_conbine[:3, ...] # [3, H, W]
    rendered_light_feat = rendered_image_conbine[3:, ...] # [LC, H, W]
    
    if spherify:
        exposure_time = torch.tensor(target_camera.exposure_time)
        ISO = torch.log(torch.tensor(target_camera.ISO * 10))
        f_number = torch.tensor(target_camera.f_number)
    else:
        exposure_time = torch.tensor(viewpoint_camera.exposure_time)
        #ISO = torch.log(torch.tensor(viewpoint_camera.ISO * 10))
        #f_number = torch.tensor(viewpoint_camera.f_number)
    if pipe.only_exposure == False:
        ISO = torch.log(torch.tensor(viewpoint_camera.ISO * 10))
        f_number = torch.tensor(viewpoint_camera.f_number)
        EV = torch.log(exposure_time) + torch.log(ISO) - 2 * torch.log(f_number)
    else:
        print("ablation: wo ISO\n")
        EV = torch.log(exposure_time)
    #print("EV", EV)
    
    if pipe.remove_lightfeature:
        print("ablation: lightfeature\n")
        lightness_map = EV
    else:
        EV = EV.expand((1, rendered_light_feat.shape[1], rendered_light_feat.shape[2])).to(rendered_light_feat.device)
        lightness_map = pc.lightmapper(EV, rendered_light_feat)
    
    rendered_image_hdr_relight = rendered_image_hdr + lightness_map
    rendered_image_sRGB = pc.tonemapper(rendered_image_hdr_relight)
    hdr = None
    if render_hdr:
        hdr = torch.exp(rendered_image_hdr)
    lightup_image = None
    if render_lightup:
        lightup_image = pc.tonemapper(rendered_image_hdr)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image_sRGB,
            "lightness_map": torch.exp(lightness_map),
            "hdr": hdr,
            "lightup": lightup_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "pixels": pixels}