"""Render a model trained with trainv2.py, with its exposure map applied.

The stock render.py knows nothing about exposure and would write the
exposure-free colours, which are not comparable with the photographs.  This
script writes, for the train and test cameras:

    <model>/<split>/ours_<iter>/renders              render x exposure map
    <model>/<split>/ours_<iter>/gt                   photographs
    <model>/<split>/ours_<iter>/exposure_maps        the exposure map itself
    <model>/<split>/ours_<iter>/renders_exposure_free

"renders" and "gt" are laid out exactly as the stock script writes them, so the
stock metrics.py works on the result unchanged:

    python render_exposure.py -m <model path>
    python metrics.py -m <model path>
"""

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import safe_state

from exposure.exposure_gaussians import ExposureGaussianModel
from exposure.exposure_render import render_with_exposure


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, use_exposure):
    base = os.path.join(model_path, name, "ours_{}".format(iteration))
    paths = {key: os.path.join(base, key)
             for key in ("renders", "gt", "exposure_maps", "renders_exposure_free")}
    for path in paths.values():
        makedirs(path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pkg = render_with_exposure(view, gaussians, pipeline, background, render,
                                   use_exposure=use_exposure)
        gt = view.original_image[0:3, :, :]
        filename = '{0:05d}'.format(idx) + ".png"
        torchvision.utils.save_image(pkg["image"], os.path.join(paths["renders"], filename))
        torchvision.utils.save_image(gt, os.path.join(paths["gt"], filename))
        torchvision.utils.save_image(pkg["exposure_map"], os.path.join(paths["exposure_maps"], filename))
        torchvision.utils.save_image(pkg["render"], os.path.join(paths["renders_exposure_free"], filename))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test):
    with torch.no_grad():
        gaussians = ExposureGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        exposure_file = os.path.join(dataset.model_path, "point_cloud",
                                     "iteration_{}".format(scene.loaded_iter), "exposures.pt")
        use_exposure = os.path.isfile(exposure_file)
        if use_exposure:
            gaussians.load_exposure(exposure_file)
            if gaussians._exposure.shape[0] != gaussians.get_xyz.shape[0]:
                raise RuntimeError("exposures.pt does not match point_cloud.ply")
        else:
            print("No exposures.pt next to the point cloud (trained with --no_exposure?): "
                  "rendering without an exposure map.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, use_exposure)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, use_exposure)


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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test)
