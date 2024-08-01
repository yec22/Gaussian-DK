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
import numpy as np
from scene import Scene
import os, time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.pose_utils import *
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import copy
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def render_fps(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    time_start = time.perf_counter()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        _ = render(view, gaussians, pipeline, background)["render"]
    time_end = time.perf_counter()
    time_consume = time_end - time_start
    img_num = idx + 1

    print('image number:', img_num)
    print(f"FPS: {img_num / time_consume:.3f}")

# xy circular 
def render_circular_video(model_path, iteration, views, gaussians, pipeline, background, radius=1.0, n_frames=240): 
    render_path = os.path.join(model_path, 'circular', "ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    # view = views[0]
    for idx in tqdm(range(n_frames)):
        view = copy.deepcopy(views[0])
        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        rendering = render(cam, gaussians, pipeline, background, target_camera = view, spherify = True)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    for idx, pose in enumerate(tqdm(generate_ellipse_path(views,n_frames=600), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    lightness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lightness")
    hdr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hdr")
    lightup_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lightup_image")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(lightness_path, exist_ok=True)
    makedirs(hdr_path, exist_ok=True)
    makedirs(lightup_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_output = render(view, gaussians, pipeline, background, render_hdr=True, render_lightup=True)
        rendering = render_output["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        lightness_map = render_output["lightness_map"] # [1, H, W]
        lightness_map = lightness_map.permute(1, 2, 0).cpu().numpy() # [H, W, 1]
        cv2.imwrite(os.path.join(lightness_path, '{0:05d}'.format(idx) + ".exr"), lightness_map)
        hdr_image = render_output["hdr"] # [3, H, W]
        hdr_image = hdr_image.permute(1, 2, 0).cpu().numpy() # [H, W, 3]
        cv2.imwrite(os.path.join(hdr_path, '{0:05d}'.format(idx) + ".exr"), hdr_image[..., ::-1])
        lightup_image = render_output["lightup"]
        torchvision.utils.save_image(lightup_image, os.path.join(lightup_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.lightness_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            #render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            render_circular_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        #if not skip_test:
            #render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            #render_circular_video(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def fps_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.lightness_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_fps(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_fps(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    #fps_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    