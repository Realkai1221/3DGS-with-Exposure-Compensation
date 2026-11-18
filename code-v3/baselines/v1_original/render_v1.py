"""Render a model trained with the original trainv1.py, with its exposure adjustment.

trainv1.py compares `adjust_color_with_exposure(render, exposure map)` with the
photograph, where the exposure map is one number per view: the mean exposure of
the Gaussians visible in it.  This script applies exactly that to the test
views, so v1 is evaluated on the picture it was trained to match.  It writes

    <model>/<split>/ours_<iter>/renders                 adjusted render
    <model>/<split>/ours_<iter>/gt                      photographs
    <model>/<split>/ours_<iter>/renders_exposure_free   what the stock render.py gives

in the layout the stock metrics.py expects.

trainv1.py itself does not save its exposure values; the notebook adds one line
to it that writes them next to the point cloud (exposure_model.pt).
"""

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

from exposure.exposure_factors import ExposureModel
from exposure.exposure_adjustment import adjust_color_with_exposure


def render_set(model_path, name, iteration, views, gaussians, exposure_model, pipeline, background):
    base = os.path.join(model_path, name, "ours_{}".format(iteration))
    paths = {key: os.path.join(base, key) for key in ("renders", "gt", "renders_exposure_free")}
    for path in paths.values():
        makedirs(path, exist_ok=True)

    factors_used = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pkg = render(view, gaussians, pipeline, background)
        image = pkg["render"]
        # the same steps as in the training loop of trainv1.py
        visible_indices = torch.where(pkg["visibility_filter"])[0]
        exposure_factors = exposure_model(visible_indices)
        H, W = image.shape[1], image.shape[2]
        exposure_map = torch.ones((1, 1, H, W), device='cuda') * exposure_factors.mean()
        image_adjusted = adjust_color_with_exposure(image.unsqueeze(0), exposure_map).squeeze(0)
        factors_used.append(exposure_factors.mean().item())

        filename = '{0:05d}'.format(idx) + ".png"
        torchvision.utils.save_image(image_adjusted, os.path.join(paths["renders"], filename))
        torchvision.utils.save_image(view.original_image[0:3, :, :], os.path.join(paths["gt"], filename))
        torchvision.utils.save_image(image, os.path.join(paths["renders_exposure_free"], filename))
    if factors_used:
        print("Mean visible exposure per view: min {:.4f}, mean {:.4f}, max {:.4f}".format(
            min(factors_used), sum(factors_used) / len(factors_used), max(factors_used)))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        exposure_file = os.path.join(dataset.model_path, "point_cloud",
                                     "iteration_{}".format(scene.loaded_iter), "exposure_model.pt")
        if not os.path.isfile(exposure_file):
            raise RuntimeError("No exposure_model.pt next to the point cloud: " + exposure_file)
        state = torch.load(exposure_file, map_location="cpu")
        exposure_model = ExposureModel(state["exposure"].shape[0])
        exposure_model.load_state_dict(state)
        exposure_model = exposure_model.cuda()
        n = gaussians.get_xyz.shape[0]
        if state["exposure"].shape[0] != n:
            raise RuntimeError("exposure_model.pt has {} values for {} Gaussians".format(
                state["exposure"].shape[0], n))
        e = exposure_model.get_all()
        print("Exposure values: min {:.4f}, mean {:.4f}, max {:.4f}".format(
            e.min().item(), e.mean().item(), e.max().item()))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, exposure_model, pipeline, background)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, exposure_model, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test)
