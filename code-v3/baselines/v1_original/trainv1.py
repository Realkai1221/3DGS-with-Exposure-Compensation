import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from exposure.exposure_factors import ExposureModel
from exposure.exposure_adjustment import adjust_color_with_exposure, adjust_depth_with_exposure
from exposure.exposure_init_call import init_exposure_supervision
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Initialize exposure model
    num_gaussians = gaussians.get_xyz.shape[0]
    exposure_model = ExposureModel(num_gaussians, init_value=0.8)
    exposure_model = exposure_model.cuda()
    
    # Create separate optimizer for exposure
    exposure_optimizer = torch.optim.Adam(exposure_model.parameters(), lr=0.001)
    
    # Initialize neighbor indices 
    def get_neighbor_indices(gaussians, k=10):
        N = gaussians.get_xyz.shape[0]
        return [[] for _ in range(N)]

    neighbor_idx = get_neighbor_indices(gaussians)
    multiview_exposure_dict = {}
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # Exposure loss weights
    lambda_weights = {
        'rgb': 1.0,
        'smooth': 0.0,
        'range': 0.01,
        'consistency': 0.0 
    }
    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            # Update neighbors after densification
            neighbor_idx = get_neighbor_indices(gaussians)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        iter_time_start = time.time()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Get visible gaussian indices
        visible_indices = torch.where(visibility_filter)[0]

        
        # Get exposure factors for visible gaussians
        exposure_factors = exposure_model(visible_indices)

        
        # Create exposure map 
        H, W = image.shape[1], image.shape[2]
        exposure_map = torch.ones((1, 1, H, W), device='cuda') * exposure_factors.mean()


        # Ground truth
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Apply exposure adjustment
        image_adjusted = adjust_color_with_exposure(
            image.unsqueeze(0), 
            exposure_map
        ).squeeze(0)

        
        # Update multiview exposure tracking
        multiview_exposure_dict = {}
        
        # Compute exposure losses
        exposure_loss = torch.tensor(0.0, device='cuda')
        loss_dict = {'L_rgb': 0.0, 'L_smooth': 0.0, 'L_range': 0.0, 'L_consistency': 0.0}


        
        # Original losses
        Ll1 = l1_loss(image_adjusted, gt_image)
        ssim_loss = (1.0 - ssim(image_adjusted, gt_image))
        
        # Total loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + exposure_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log exposure losses
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('exposure/L_rgb', loss_dict['L_rgb'], iteration)
                tb_writer.add_scalar('exposure/L_smooth', loss_dict['L_smooth'], iteration)
                tb_writer.add_scalar('exposure/L_range', loss_dict['L_range'], iteration)
                tb_writer.add_scalar('exposure/L_consistency', loss_dict['L_consistency'], iteration)

    
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Before densification, save current exposure values
                    old_exposure = exposure_model.exposure.data.clone()
                    
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    # After densification, update exposure model size
                    new_num_gaussians = gaussians.get_xyz.shape[0]
                    if new_num_gaussians != num_gaussians:
                        new_exposure = torch.ones(new_num_gaussians, device='cuda') * 0.8
                        new_exposure[:min(num_gaussians, new_num_gaussians)] = old_exposure[:min(num_gaussians, new_num_gaussians)]
                        exposure_model.exposure = torch.nn.Parameter(new_exposure)
                        exposure_optimizer = torch.optim.Adam(exposure_model.parameters(), lr=0.001)
                        num_gaussians = new_num_gaussians
                        neighbor_idx = get_neighbor_indices(gaussians)

            # Save PLY files at specified iterations
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")

                # Create the correct directory structure
                point_cloud_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
                os.makedirs(point_cloud_path, exist_ok=True)

                # Save PLY file directly
                ply_file_path = os.path.join(point_cloud_path, "point_cloud.ply")
                gaussians.save_ply(ply_file_path)

                print(f"Saved PLY to: {ply_file_path}")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args, opt=None, pipe=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "cfg_args_opt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))

    with open(os.path.join(args.model_path, "cfg_args_pipe"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
