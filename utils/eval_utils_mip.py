import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer_mip import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
import glob

matplotlib.use("Agg")
def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def evaluate_evo_all(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("ALL RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_all_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_all_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate



def eval_ate_all(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = frames[len(frames) - 1].uid
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for id in range(len(frames)):
        kf = frames[id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_all_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo_all(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"ate all frame , frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    skipstep = 1,
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[skipstep*idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)   #RGB
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(   #RGB
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)   #BGR
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)   #BGR
        if idx == 0 or idx == int(end_idx/5) or idx == int(end_idx*2/5) or idx == int(3*end_idx/5) or idx == int(4*end_idx/5):
            cv2.imshow("gt_image" , gt) ; cv2.imshow("pred_image" , pred) ; cv2.imshow("diffimage" , np.abs(pred-gt))  ; cv2.waitKey(0) ; cv2.destroyAllWindows()
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
        print(psnr_score , ssim_score , lpips_score)
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output



def eval_rendering_vings(
    frames,
    input_folder,
    gaussians,
    save_dir,
    pipe,
    background,
    kf_indices,
    map1x, map1y,
    iteration="final",
    deltatime = 0.01
):
    color_paths = sorted(
            glob.glob(f"{input_folder}/mav0/cam0/data/*.png")
        )
    gtimage_timestamps = [ float(color_path.split("/")[-1].split(".")[0])/1e9 for color_path in color_paths]
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    gttimestamp_id = 0

    for idx in range(0, end_idx):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        while frame.timestamp - gtimage_timestamps[gttimestamp_id] > deltatime:
            gttimestamp_id += 1
        assert(abs(frame.timestamp - gtimage_timestamps[gttimestamp_id]) < deltatime)
        gt_image = cv2.imread(color_paths[gttimestamp_id])
        
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)  # 转换颜色格式为 RGB
        gt_image = cv2.remap(gt_image, map1x, map1y, cv2.INTER_LINEAR)
        gt_image = torch.from_numpy(gt_image).to("cuda").float() / 255.0  # 转换为 PyTorch 张量并归一化
        gt_image = gt_image.permute(2, 0, 1)  # transfer to K opt

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)  #BGR
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB) #BGR
        #cv2.imshow("gt_image" , gt) ; cv2.imshow("pred_image" , pred) ; cv2.waitKey(0) ; cv2.destroyAllWindows() 
        if idx == 0 or idx == int(end_idx/5) or idx == int(end_idx*2/5) or idx == int(3*end_idx/5) or idx == int(4*end_idx/5):
            cv2.imshow("gt_image" , gt) ; cv2.imshow("pred_image" , pred) ; cv2.imshow("diffimage" , np.abs(pred-gt))  ; cv2.waitKey(0) ; cv2.destroyAllWindows()
            print(psnr_score , ssim_score , lpips_score)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        #print(psnr_score , ssim_score , lpips_score)
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output



def eval_rendering_vings_mip(
    frames,
    input_folder,
    gaussians,
    save_dir,
    pipe,
    background,
    kf_indices,
    map1x, map1y,
    iteration="final",
    deltatime = 0.01,
    kernel_size = 0.1
):
    color_paths = sorted(
            glob.glob(f"{input_folder}/mav0/cam0/data/*.png")
        )
    gtimage_timestamps = [ float(color_path.split("/")[-1].split(".")[0])/1e9 for color_path in color_paths]
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    gttimestamp_id = 0

    for idx in range(0, end_idx):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        while frame.timestamp - gtimage_timestamps[gttimestamp_id] > deltatime:
            gttimestamp_id += 1
        assert(abs(frame.timestamp - gtimage_timestamps[gttimestamp_id]) < deltatime)
        gt_image = cv2.imread(color_paths[gttimestamp_id])
        
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)  # 转换颜色格式为 RGB
        gt_image = cv2.remap(gt_image, map1x, map1y, cv2.INTER_LINEAR)
        gt_image = torch.from_numpy(gt_image).to("cuda").float() / 255.0  # 转换为 PyTorch 张量并归一化
        gt_image = gt_image.permute(2, 0, 1)  # transfer to K opt

        rendering = render(frame, gaussians, pipe, background,kernel_size=kernel_size)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)  #BGR
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB) #BGR
        #cv2.imshow("gt_image" , gt) ; cv2.imshow("pred_image" , pred) ; cv2.waitKey(0) ; cv2.destroyAllWindows() 
        if idx == 0 or idx == int(end_idx/5) or idx == int(end_idx*2/5) or idx == int(3*end_idx/5) or idx == int(4*end_idx/5):
            cv2.imshow("gt_image" , gt) ; cv2.imshow("pred_image" , pred) ; cv2.imshow("diffimage" , np.abs(pred-gt))  ; cv2.waitKey(0) ; cv2.destroyAllWindows()
            print(psnr_score , ssim_score , lpips_score)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        #print(psnr_score , ssim_score , lpips_score)
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
