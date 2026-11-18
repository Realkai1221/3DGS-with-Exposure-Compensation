"""3DGS training with per-Gaussian exposure compensation, version 3.

Same method as trainv2.py, with the parts that did not do what the paper
describes repaired.  See README.md for the list; in short:

  1. exposures are initialised from the grey level each Gaussian projects to
  2. new Gaussians inherit their parent's exposure (exposure is stored in the
     Gaussian model, so pruning and densification keep it aligned)
  3. the consistency loss has a gradient
  4. the smoothness loss runs every iteration at any scene size
  5. the exposure map is rendered from the per-Gaussian exposures
  6. densification, opacity reset and gradient clipping are back to stock 3DGS
     by default, each behind a flag so its effect can be measured

Drop this file and the exposure/ folder into the root of the 2023
gaussian-splatting repository and run it like train.py.
"""

import os
import shutil
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

from exposure.exposure_factors import initialize_exposure_from_projection
from exposure.exposure_gaussians import ExposureGaussianModel
from exposure.exposure_init_call import init_exposure_supervision
from exposure.exposure_render import render_with_exposure

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def exposure_lr_at(iteration, base_lr):
    """Same schedule as v2: a tenth of the rate for 3,000 iterations, then the
    full rate, decayed by 2% every 2,000 iterations."""
    if iteration < 3000:
        return base_lr * 0.1
    return base_lr * (0.98 ** ((iteration - 3000) // 2000))


def load_checkpoint(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:                      # PyTorch older than 1.13 has no such argument
        return torch.load(path)


def save_checkpoint(gaussians, iteration, model_path, copy_dir=None):
    """Write a checkpoint training can resume from.  With `copy_dir` it is also
    copied there as checkpoint.pth (e.g. a mounted drive that outlives the machine)."""
    state = gaussians.capture_exposure()
    state['gaussians'] = gaussians.capture()
    state['iteration'] = iteration
    path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    torch.save(state, path)
    if not copy_dir:
        return
    try:
        os.makedirs(copy_dir, exist_ok=True)
        size, free = os.path.getsize(path), shutil.disk_usage(copy_dir).free
        if free < size + 2e9:
            print(f"Checkpoint not copied to {copy_dir}: {size / 1e9:.1f} GB needed, "
                  f"{free / 1e9:.1f} GB free")
            return
        target = os.path.join(copy_dir, "checkpoint.pth")
        shutil.copyfile(path, target + ".part")
        os.replace(target + ".part", target)
        print(f"Checkpoint copied to {target} ({size / 1e9:.1f} GB)")
    except Exception as exc:               # never stop training because a copy failed
        print(f"Checkpoint not copied to {copy_dir}: {exc}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, ex):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe, ex)
    use_exposure = not ex.no_exposure

    gaussians = ExposureGaussianModel(dataset.sh_degree,
                                      exposure_init_value=ex.exposure_init_value,
                                      history_window=ex.history_window)
    scene = Scene(dataset, gaussians)
    gaussians.exposure_lr = exposure_lr_at(1, ex.exposure_lr)

    if checkpoint:
        checkpoint_data = load_checkpoint(checkpoint)
        if not isinstance(checkpoint_data, dict) or 'exposure' not in checkpoint_data:
            sys.exit("This checkpoint was not written by trainv3.py and cannot be resumed with it.")
        gaussians.set_exposure_tensor(checkpoint_data['exposure'])
        gaussians.restore(checkpoint_data['gaussians'], opt)
        gaussians.restore_exposure_history(checkpoint_data)
        first_iter = checkpoint_data['iteration']
        print(f"Resumed from {checkpoint} at iteration {first_iter} "
              f"with {gaussians.get_xyz.shape[0]} Gaussians")
    else:
        if use_exposure:
            if ex.exposure_init == "projection":
                e_init, seen = initialize_exposure_from_projection(
                    gaussians.get_xyz, scene.getTrainCameras(), default=ex.exposure_init_value)
                gaussians.initialise_exposure(e_init)
                print(f"Exposure initialised from projection: {int(seen.sum())}/{seen.numel()} "
                      f"Gaussians seen, mean {e_init.mean().item():.3f}, "
                      f"min {e_init.min().item():.3f}, max {e_init.max().item():.3f}")
            else:
                gaussians.initialise_exposure()
                print(f"Exposure initialised to the constant {ex.exposure_init_value}")
        gaussians.training_setup(opt)

    print("\n=== Densification settings ===")
    print(f"densify_grad_threshold: {opt.densify_grad_threshold} x {ex.densify_scale}")
    print(f"densification_interval: {opt.densification_interval}")
    print(f"densify_from_iter / until_iter: {opt.densify_from_iter} / {opt.densify_until_iter}")
    print(f"opacity reset: {'off' if ex.no_opacity_reset else 'every %d' % opt.opacity_reset_interval}")
    print(f"exposure: {'off' if ex.no_exposure else 'on'}")
    print("==============================\n")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    lambda_weights = {
        'rgb': ex.lambda_exp_rgb,
        'smooth': ex.lambda_smooth,
        'range': ex.lambda_range,
        'consistency': ex.lambda_consistency,
    }
    # During warm-up only the RGB term is used, as in v2.
    warmup_weights = {'rgb': ex.lambda_exp_rgb, 'smooth': 0.0, 'range': 0.0, 'consistency': 0.0}
    use_smooth = use_exposure and ex.lambda_smooth > 0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.set_exposure_lr(exposure_lr_at(iteration, ex.exposure_lr))

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Neighbour graph for the smoothness loss.  Between rebuilds it is kept
        # consistent through densification by the Gaussian model itself.
        if use_smooth and iteration > ex.exposure_warmup and \
                (gaussians.knn_idx is None or iteration % ex.knn_interval == 0):
            gaussians.rebuild_neighbors(ex.knn_k)

        # Render the exposure-free image and the exposure map, and combine them
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render_with_exposure(viewpoint_cam, gaussians, pipe, background, render,
                                          use_exposure=use_exposure)
        image = render_pkg["image"]
        exposure_map = render_pkg["exposure_map"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        visible_idx = torch.nonzero(visibility_filter).squeeze(1)

        gt_image = viewpoint_cam.original_image.cuda()

        # Original losses, on the image with exposure applied
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Exposure losses
        loss_dict = {'L_rgb': 0.0, 'L_smooth': 0.0, 'L_range': 0.0, 'L_consistency': 0.0}
        if use_exposure:
            weights = lambda_weights if iteration > ex.exposure_warmup else warmup_weights
            exposure_loss, loss_dict = init_exposure_supervision(
                image, gt_image, gaussians, visible_idx, weights, ex.smooth_samples)
            exposure_weight = min(1.0, iteration / 5000.0)
            loss = loss + exposure_weight * exposure_loss

        loss.backward()

        if ex.clip_xyz_grad:
            torch.nn.utils.clip_grad_norm_(gaussians._xyz, max_norm=1.0)
        if use_exposure and gaussians._exposure.grad is not None:
            torch.nn.utils.clip_grad_norm_([gaussians._exposure], max_norm=1.0)

        iter_end.record()

        with torch.no_grad():
            if use_exposure:
                gaussians.update_exposure_history(visible_idx)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Exp": f"{exposure_map.mean().item():.3f}",
                    "N": gaussians.get_xyz.shape[0],
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log exposure losses and statistics
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('exposure/L_rgb', loss_dict['L_rgb'], iteration)
                tb_writer.add_scalar('exposure/L_smooth', loss_dict['L_smooth'], iteration)
                tb_writer.add_scalar('exposure/L_range', loss_dict['L_range'], iteration)
                tb_writer.add_scalar('exposure/L_consistency', loss_dict['L_consistency'], iteration)
                tb_writer.add_scalar('exposure/mean_value', gaussians.get_exposure.mean().item(), iteration)
                tb_writer.add_scalar('exposure/std_value', gaussians.get_exposure.std().item(), iteration)
                tb_writer.add_scalar('exposure/map_std', exposure_map.std().item(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, gaussians, pipe, background, use_exposure)

            # Save the model.  As in stock 3DGS this comes before densification, so the
            # saved model is the one that was just evaluated.
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                point_cloud_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
                os.makedirs(point_cloud_path, exist_ok=True)
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                if use_exposure:
                    gaussians.save_exposure(os.path.join(point_cloud_path, "exposures.pt"), iteration)
                    gaussians.save_baked_ply(os.path.join(point_cloud_path, "point_cloud_baked.ply"))
                print(f"Saved to: {point_cloud_path}")

            # Densification (stock 3DGS; the model carries the exposures along)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    n_before = gaussians.get_xyz.shape[0]
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold * ex.densify_scale, 0.005,
                                                scene.cameras_extent, size_threshold)
                    if iteration % 1000 == 0:
                        print(f"\n[ITER {iteration}] Densification: {n_before} -> "
                              f"{gaussians.get_xyz.shape[0]} gaussians")

                # v1/v2 left this out.  It is the step stock 3DGS relies on to
                # remove floaters, so it is on unless --no_opacity_reset is given.
                if not ex.no_opacity_reset and (
                        iteration % opt.opacity_reset_interval == 0 or
                        (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # Optimizer step (the exposures are a parameter group of this optimizer)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # As in stock 3DGS the checkpoint is written after the step, so a
            # resumed run continues exactly where this one was.
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_checkpoint(gaussians, iteration, scene.model_path, ex.checkpoint_copy_dir)


def prepare_output_and_logger(args, opt=None, pipe=None, ex=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if opt:
        with open(os.path.join(args.model_path, "cfg_args_opt"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(opt))))

    if pipe:
        with open(os.path.join(args.model_path, "cfg_args_pipe"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(pipe))))

    if ex:
        with open(os.path.join(args.model_path, "cfg_args_exposure"), 'w') as cfg_log_f:
            cfg_log_f.write(str(ex))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene: Scene,
                    gaussians, pipe, background, use_exposure):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                           for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Evaluate the same picture that was trained: render x exposure map.
                    pkg = render_with_exposure(viewpoint, gaussians, pipe, background, render,
                                               use_exposure=use_exposure)
                    image = torch.clamp(pkg["image"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/exposure_map".format(viewpoint.image_name),
                                             torch.clamp(pkg["exposure_map"], 0.0, 1.0)[None],
                                             global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/exposure_histogram", gaussians.get_exposure, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_copy_dir", type=str, default=None,
                        help="also copy each checkpoint to this folder as checkpoint.pth")

    # Exposure compensation
    eg = parser.add_argument_group("Exposure compensation")
    eg.add_argument("--no_exposure", action="store_true",
                    help="exposure map fixed to 1: plain 3DGS through this script (baseline)")
    eg.add_argument("--exposure_init", choices=["projection", "constant"], default="projection",
                    help="start each exposure from its projected grey level, or from a constant")
    eg.add_argument("--exposure_init_value", type=float, default=0.95,
                    help="constant start value (also used for Gaussians seen by no camera)")
    eg.add_argument("--exposure_lr", type=float, default=0.001)
    eg.add_argument("--exposure_warmup", type=int, default=1000,
                    help="iterations with only the RGB exposure loss")
    eg.add_argument("--lambda_exp_rgb", type=float, default=1.0)
    eg.add_argument("--lambda_smooth", type=float, default=0.1)
    eg.add_argument("--lambda_range", type=float, default=0.01)
    eg.add_argument("--lambda_consistency", type=float, default=0.05)
    eg.add_argument("--knn_k", type=int, default=8)
    eg.add_argument("--knn_interval", type=int, default=1000,
                    help="iterations between full rebuilds of the neighbour graph")
    eg.add_argument("--smooth_samples", type=int, default=50000,
                    help="visible Gaussians used for the smoothness loss each iteration")
    eg.add_argument("--history_window", type=int, default=10,
                    help="views remembered per Gaussian for the consistency loss")

    # Differences from stock 3DGS that v2 had hard-coded.  Defaults are stock.
    ag = parser.add_argument_group("Ablation switches (defaults = stock 3DGS)")
    ag.add_argument("--densify_scale", type=float, default=1.0,
                    help="multiplier on densify_grad_threshold (v2 used 5.0)")
    ag.add_argument("--clip_xyz_grad", action="store_true",
                    help="clip the position gradient norm to 1 (v2 did)")
    ag.add_argument("--no_opacity_reset", action="store_true",
                    help="skip the periodic opacity reset (v1 and v2 did)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    exposure_args = Namespace(**{k: getattr(args, k) for k in (
        "no_exposure", "exposure_init", "exposure_init_value", "exposure_lr", "exposure_warmup",
        "lambda_exp_rgb", "lambda_smooth", "lambda_range", "lambda_consistency",
        "knn_k", "knn_interval", "smooth_samples", "history_window",
        "densify_scale", "clip_xyz_grad", "no_opacity_reset", "checkpoint_copy_dir")})

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from, exposure_args)

    # All done
    print("\nTraining complete.")
