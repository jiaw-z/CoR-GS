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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torchvision
import numpy as np
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, loss_photometric
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import random

from utils.visualization_utils import depth2image, visualize_cmap

import kmeans1d
import open3d as o3d

import copy

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def training(dataset, opt, pipe, args):
    # implenmetation of more than 2 3d gaussian radiance fields currently is not supported in this code
    assert args.gaussiansN >= 1 and args.gaussiansN <=2 
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    print(f"scene.bounds is {scene.bounds}")
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    GsDict = {}
    for i in range(args.gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        elif i > 0:
            GsDict[f"gs{i}"] = GaussianModel(args)
            GsDict[f"gs{i}"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    pseudo_stack_co = None

    allCameras = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            for i in range(args.gaussiansN):
                GsDict[f"gs{i}"].oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        if 'DTU' in scene.source_path:
            if 'scan110' not in scene.source_path:
                bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3,1,1)] = 0.
        else:
            bg_mask = None

        RenderDict = {}
        LossDict = {}
        logDict = {}

        # render for main viewpoint
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        for i in range(args.gaussiansN):
            RenderDict[f"render_pkg_gs{i}"] = render(viewpoint_cam, GsDict[f'gs{i}'], pipe, bg)
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"depth_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["depth"]
            RenderDict[f"alpha_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["alpha"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]

        # Loss
        for i in range(args.gaussiansN):
            if not bg_mask is None:
                LossDict[f"loss_gs{i}"] = loss_photometric(RenderDict[f"image_gs{i}"], gt_image, opt=opt, valid=(~bg_mask).float())
                LossDict[f"loss_gs{i}"] += (RenderDict[f"alpha_gs{i}"][bg_mask]**2).mean()
            else:
                LossDict[f"loss_gs{i}"] = loss_photometric(RenderDict[f"image_gs{i}"], gt_image, opt=opt)

        if not args.onlyrgb:
            if iteration % args.sample_pseudo_interval == 0 and iteration <= args.end_sample_pseudo:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                if not pseudo_stack_co:
                    pseudo_stack_co = scene.getPseudoCameras().copy()
                pseudo_cam_co = pseudo_stack_co.pop(randint(0, len(pseudo_stack_co) - 1))

                for i in range(args.gaussiansN):
                        RenderDict[f"render_pkg_pseudo_co_gs{i}"] = render(pseudo_cam_co, GsDict[f'gs{i}'], pipe, bg)
                        RenderDict[f"image_pseudo_co_gs{i}"] = RenderDict[f"render_pkg_pseudo_co_gs{i}"]["render"]
                        RenderDict[f"depth_pseudo_co_gs{i}"] = RenderDict[f"render_pkg_pseudo_co_gs{i}"]["depth"]
                if iteration >= args.start_sample_pseudo:
                    ####################################################################
                    # co-reg
                    if args.coreg:
                        # co photometric
                        for i in range(args.gaussiansN):
                            for j in range(args.gaussiansN):
                                if i != j:
                                    LossDict[f"loss_gs{i}"] += loss_photometric(RenderDict[f"image_pseudo_co_gs{i}"], RenderDict[f"image_pseudo_co_gs{j}"].clone().detach(), opt=opt) / (args.gaussiansN - 1)

            


        loss = LossDict["loss_gs0"]
        for i in range(args.gaussiansN):
            LossDict[f"loss_gs{i}"].backward()

        if args.save_log_images and (iteration % 100 == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                render_results = render(eval_cam, GsDict[f'gs0'], pipe, bg)
                image = torch.clamp(render_results["render"], 0.0, 1.0)
                gt_image = torch.clamp(eval_cam.original_image.to("cuda"), 0.0, 1.0)
                black = torch.zeros_like(gt_image).to(gt_image.device)
                render_depth = render_results["depth"]
                render_depth_image = depth2image(render_depth, inverse=True, rgb=True)
                render_opacity_image = render_results["alpha"].repeat(3, 1, 1)

                if args.gaussiansN > 1:
                    render_results_gs1 = render(eval_cam, GsDict[f'gs1'], pipe, bg)
                    image_gs1 = torch.clamp(render_results_gs1["render"], 0.0, 1.0)
                    render_depth_gs1 = render_results_gs1["depth"]
                    render_depth_image_gs1 = depth2image(render_depth_gs1, inverse=True, rgb=True)
                    render_opacity_image_gs1 = render_results_gs1["alpha"].repeat(3, 1, 1)

            row0 = torch.cat([gt_image, black, black], dim=2)
            row1 = torch.cat([image, render_depth_image, render_opacity_image], dim=2)
            if args.gaussiansN > 1:
                row2 = torch.cat([image_gs1, render_depth_image_gs1, render_opacity_image_gs1], dim=2)
            else:
                row2 = torch.cat([black, black, black], dim=2)

            image_to_show = torch.cat([row0, row1, row2], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images_train", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images_train/{iteration}.jpg")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(args, tb_writer, iteration, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background), GsDict=GsDict)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if args.gaussiansN == 2:
                    pcd_path = os.path.join(scene.model_path, "point_cloud_gs2/iteration_{}".format(iteration))
                    GsDict["gs1"].save_ply(os.path.join(pcd_path, "point_cloud.ply"))

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((GsDict["gs0"].capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                for i in range(args.gaussiansN):
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter])
                    GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                # density and prune
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    for i in range(args.gaussiansN):

                        GsDict[f"gs{i}"].densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)                              

            # Optimizer step
            if iteration < opt.iterations:
                for i in range(args.gaussiansN):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none = True)

            for i in range(args.gaussiansN):
                GsDict[f"gs{i}"].update_learning_rate(iteration)
                if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                        iteration > args.start_sample_pseudo:
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"reset opacity of gaussians-{i} at iteration {iteration}")
                    GsDict[f"gs{i}"].reset_opacity()
                    
            if args.coprune and iteration > opt.densify_from_iter and iteration % 500 == 0:
                for i in range(args.gaussiansN):
                    for j in range(args.gaussiansN):
                        if i != j:
                            source_cloud = o3d.geometry.PointCloud()
                            source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_xyz.clone().cpu().numpy())
                            target_cloud = o3d.geometry.PointCloud()
                            target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{j}"].get_xyz.clone().cpu().numpy())
                            trans_matrix = np.identity(4)
                            threshold = args.coprune_threshold
                            evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_matrix)
                            correspondence = np.array(evaluation.correspondence_set)
                            mask_consistent = torch.zeros((GsDict[f"gs{i}"].get_xyz.shape[0], 1)).cuda()
                            mask_consistent[correspondence[:, 0], :] = 1
                            GsDict[f"indice_consistent_gs{i}to{j}"] = correspondence
                            GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())
                for i in range(args.gaussiansN):
                    GsDict[f"gs{i}"].prune_from_mask(GsDict[f"mask_inconsistent_gs{i}"].squeeze(), iter=iteration)
                    


def prepare_output_and_logger(args):
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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(args, tb_writer, iteration, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, GsDict=None):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                
    if 'DTU' in scene.source_path:
        depth_rgb = True
    else:
        depth_rgb = True
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                MetricDict = {}
                for i in range(args.gaussiansN):
                    if i != 0:
                        MetricDict[f"l1_test_gs{i}"], MetricDict[f"psnr_test_gs{i}"], MetricDict[f"ssim_test_gs{i}"], MetricDict[f"lpips_test_gs{i}"] = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)   
                    black = torch.zeros_like(gt_image).to(gt_image.device) 
                    RenderResults = {}
                    
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    render_image = torch.clamp(render_results["render"], 0.0, 1.0)
                    render_depth = render_results["depth"]
                    render_depth_image = depth2image(render_depth, inverse=True, rgb=depth_rgb)
                    render_opacity_image = render_results["alpha"].repeat(3, 1, 1)

                    if args.gaussiansN > 1:
                        render_results_gs1 = renderFunc(viewpoint, GsDict['gs1'], *renderArgs)
                        render_image_gs1 = torch.clamp(render_results_gs1["render"], 0.0, 1.0)
                        render_depth_gs1 = render_results_gs1["depth"]
                        render_depth_image_gs1 = depth2image(render_depth_gs1, inverse=True, rgb=depth_rgb)
                        render_opacity_image_gs1 = render_results_gs1["alpha"].repeat(3, 1, 1)

                               


                    if tb_writer and (idx < 8):
                        row0 = torch.cat([gt_image, black, black], dim=2)
                        row1 = torch.cat([render_image, render_depth_image, render_opacity_image], dim=2)
                        if args.gaussiansN > 1:
                            row2 = torch.cat([render_image_gs1, render_depth_image_gs1, render_opacity_image_gs1], dim=2)
                        else:
                            row2 = torch.cat([black, black, black], dim=2)
                        
                        image_to_show = torch.cat([row0, row1, row2], dim=1)
                        image_to_show = torch.clamp(image_to_show, 0, 1)
                        
                        save_path = f"{args.model_path}/save_images_{config['name']}/view_{viewpoint.image_name}"
                        os.makedirs(save_path, exist_ok = True)
                        torchvision.utils.save_image(image_to_show, save_path + f"/{iteration}.jpg") 

                        tb_writer.add_images(config['name'] + "_view_{}/render_image".format(viewpoint.image_name), render_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(viewpoint.image_name), render_depth_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)


                    l1_test += l1_loss(render_image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(render_image, gt_image, _mask).mean().double()
                    _ssim = ssim(render_image, gt_image, _mask).mean().double()
                    _lpips = lpips(render_image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 2000, 3000, 5000, 7000, 10000, 15000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--checkpoint2", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")

    parser.add_argument('--gaussiansN', type=int, default=1)

    parser.add_argument("--onlyrgb", action='store_true', default=False)

    parser.add_argument("--coreg", action='store_true', default=False)
    parser.add_argument("--coprune", action='store_true', default=False)
    parser.add_argument('--coprune_threshold', type=int, default=5)

    parser.add_argument("--save_log_images", action="store_true")

    # parser.add_argument("--absdensify", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        print(f"merge configs from {args.configs}")

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    seed_everything(42)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")