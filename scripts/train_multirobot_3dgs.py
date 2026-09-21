#!/usr/bin/env python3
"""
train_multirobot_3dgs.py: Real Multi-Robot 3D Gaussian Splatting Training Pipeline.

Optimizes photorealistic 3D Gaussians per sub-sequence (Robot_A, Robot_B, Robot_D, Robot_E)
against real RGB video streams using gsplat and Adam optimization, then transforms and merges
all sub-scenes via Sim(3) matrices from transforms.json into a globally consistent, photorealistic
3DGS scene (.splat and .ply).

Usage:
    python scripts/train_multirobot_3dgs.py --scene-dir outputs/scenes/scout_scene_20260920_200503
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import gsplat

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from scripts.pcd_to_3dgs import write_3dgs_ply, write_splat_binary
except ImportError:
    from pcd_to_3dgs import write_3dgs_ply, write_splat_binary

# Constant for SH Degree 0 DC: C0 = 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814


def create_window(window_size: int = 11, channel: int = 3, device: str = "cuda") -> torch.Tensor:
    """Create 2D Gaussian window for differentiable SSIM computation."""
    def _gaussian(size: int, sigma: float):
        gauss = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2.0 * sigma ** 2))
        return gauss / gauss.sum()

    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Differentiable SSIM loss.
    img1, img2: [B, H, W, 3] in [0, 1] range.
    """
    i1 = img1.permute(0, 3, 1, 2)
    i2 = img2.permute(0, 3, 1, 2)
    channel = 3

    mu1 = F.conv2d(i1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(i2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(i1 * i1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(i2 * i2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(i1 * i2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def solve_camera_intrinsics(
    world_points_tensor: torch.Tensor,
    camera_poses: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Precompute pinhole camera intrinsics K per frame via least squares from world_points and c2w poses.
    Returns: Ks array of shape [N, 3, 3]
    """
    n_frames = len(camera_poses)
    Ks = np.zeros((n_frames, 3, 3), dtype=np.float32)

    grid_y, grid_x = np.indices((H, W))
    u = grid_x.flatten()
    v = grid_y.flatten()

    for i in range(n_frames):
        c2w = camera_poses[i]
        w2c = np.linalg.inv(c2w)
        pts_world = world_points_tensor[i].numpy().reshape(-1, 3)
        pts_cam = (pts_world @ w2c[:3, :3].T) + w2c[:3, 3]
        Z = pts_cam[:, 2]

        valid = (Z > 0.1) & (Z < 20.0) & np.isfinite(pts_cam).all(axis=1)
        if valid.sum() > 200:
            a_u = np.stack([pts_cam[valid, 0] / Z[valid], np.ones(valid.sum())], axis=-1)
            sol_u, _, _, _ = np.linalg.lstsq(a_u, u[valid], rcond=None)
            a_v = np.stack([pts_cam[valid, 1] / Z[valid], np.ones(valid.sum())], axis=-1)
            sol_v, _, _, _ = np.linalg.lstsq(a_v, v[valid], rcond=None)
            fx, cx = float(sol_u[0]), float(sol_u[1])
            fy, cy = float(sol_v[0]), float(sol_v[1])
        else:
            fx, fy, cx, cy = 450.0, 450.0, W / 2.0, H / 2.0

        Ks[i] = [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]

    return Ks


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply quaternions in (w, x, y, z) convention.
    q1: shape (4,) or (N, 4)
    q2: shape (N, 4)
    Returns: (N, 4) product q1 * q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def train_single_sequence(
    seq_name: str,
    stream_dir: Path,
    total_steps: int = 800,
    lr_means: float = 1.6e-4,
    lr_colors: float = 1e-2,
    lr_opacities: float = 5e-2,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    prune_opacity: float = 0.05,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Trains 3D Gaussians for a single robot video stream.
    Returns: Dictionary of optimized Gaussian parameters {means, scales, quats, opacities, colors}
    """
    print(f"\n=======================================================")
    print(f"[*] Training Sub-Scene: {seq_name}")
    print(f"=======================================================")

    # 1. Load Data
    pose_path = stream_dir / "camera_poses.npy"
    color_path = stream_dir / "colors.pt"
    mask_path = stream_dir / "static_masks.pt"
    wp_path = stream_dir / "world_points.pt"
    ply_path = stream_dir / "reconstruction.ply"

    if not pose_path.exists() or not color_path.exists() or not ply_path.exists():
        raise FileNotFoundError(f"Missing required data in {stream_dir}")

    camera_poses = np.load(pose_path)
    colors_raw = torch.load(color_path, map_location="cpu", weights_only=False)
    masks_raw = torch.load(mask_path, map_location="cpu", weights_only=False) if mask_path.exists() else None
    wp_raw = torch.load(wp_path, map_location="cpu", weights_only=False)

    N_frames, H, W, _ = colors_raw.shape
    print(f"  Stream frames: {N_frames} ({W}x{H} RGB)")

    # Normalize colors to [0, 1] float
    gt_colors = (colors_raw.float() / 255.0).to(device)
    if masks_raw is not None:
        gt_masks = masks_raw.float().unsqueeze(-1).to(device)
    else:
        gt_masks = torch.ones((N_frames, H, W, 1), device=device)

    # 2. Camera Intrinsics & Extrinsics
    print(f"  Estimating per-frame intrinsics via least-squares...")
    Ks = solve_camera_intrinsics(wp_raw, camera_poses, H, W)
    Ks_cu = torch.from_numpy(Ks).float().to(device)

    # w2c matrices for gsplat
    w2c_poses = np.linalg.inv(camera_poses)
    viewmats_cu = torch.from_numpy(w2c_poses).float().to(device)

    # 3. Initialize Gaussians from reconstruction.ply
    print(f"  Loading seed point cloud from {ply_path.name}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_np = np.asarray(pcd.points, dtype=np.float32)
    cols_np = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.full_like(pts_np, 0.7)
    N_gaussians = len(pts_np)
    print(f"  Initial seed Gaussians: {N_gaussians:,}")

    # Parameters
    means = torch.nn.Parameter(torch.from_numpy(pts_np).float().to(device))
    quats = torch.nn.Parameter(torch.zeros((N_gaussians, 4), device=device))
    quats.data[:, 0] = 1.0  # w=1 identity quaternion

    # Adaptive scale initialization
    init_scale = 0.007
    log_scales = torch.nn.Parameter(torch.full((N_gaussians, 3), math.log(init_scale), device=device))

    # Opacity logits initialization (init ~ 0.8)
    init_opacity = 0.8
    init_opac_logit = math.log(init_opacity / (1.0 - init_opacity))
    opacity_logits = torch.nn.Parameter(torch.full((N_gaussians,), init_opac_logit, device=device))

    # Color logits initialization (inverse sigmoid)
    cols_clamped = np.clip(cols_np, 0.001, 0.999)
    color_logits_init = np.log(cols_clamped / (1.0 - cols_clamped))
    color_logits = torch.nn.Parameter(torch.from_numpy(color_logits_init).float().to(device))

    # Scale step count for short sequences
    actual_steps = min(total_steps, max(150, N_frames * 40))
    print(f"  Optimizing for {actual_steps} steps (batch size=1, lr_means={lr_means})...")

    # Optimizer
    opt = torch.optim.Adam([
        {"params": [means], "lr": lr_means},
        {"params": [quats], "lr": lr_quats},
        {"params": [log_scales], "lr": lr_scales},
        {"params": [opacity_logits], "lr": lr_opacities},
        {"params": [color_logits], "lr": lr_colors},
    ])

    ssim_window = create_window(device=device)

    # Initial PSNR on frame 0
    with torch.no_grad():
        ren0_init, _, _ = gsplat.rasterization(
            means=means,
            quats=F.normalize(quats, dim=-1),
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            viewmats=viewmats_cu[0:1],
            Ks=Ks_cu[0:1],
            width=W,
            height=H,
            sh_degree=None,
        )
        init_mse = F.mse_loss(ren0_init[0], gt_colors[0]).item()
        init_psnr = -10.0 * math.log10(max(init_mse, 1e-8))

    # 4. Training Loop
    t_start = time.time()
    for step in range(actual_steps):
        # Sample random camera view
        idx = np.random.randint(0, N_frames)

        # Exponential decay for means learning rate
        lr_means_cur = lr_means * (0.01 ** (step / max(actual_steps, 1)))
        opt.param_groups[0]["lr"] = lr_means_cur

        scales = torch.exp(log_scales)
        opacities = torch.sigmoid(opacity_logits)
        cur_colors = torch.sigmoid(color_logits)
        norm_quats = F.normalize(quats, dim=-1)

        # Forward Rasterization
        renders, alphas, meta = gsplat.rasterization(
            means=means,
            quats=norm_quats,
            scales=scales,
            opacities=opacities,
            colors=cur_colors,
            viewmats=viewmats_cu[idx:idx+1],
            Ks=Ks_cu[idx:idx+1],
            width=W,
            height=H,
            sh_degree=None,
        )

        gt = gt_colors[idx:idx+1]
        m = gt_masks[idx:idx+1]

        # Photometric L1 Loss weighted by static mask
        l1 = (torch.abs(renders - gt) * m).sum() / (m.sum() * 3.0 + 1e-6)

        # SSIM Loss
        l_ssim = 1.0 - compute_ssim(renders, gt, ssim_window)

        # Composite Loss: 0.8 L1 + 0.2 D-SSIM
        loss = 0.8 * l1 + 0.2 * l_ssim

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0 or (step + 1) == actual_steps:
            elapsed = time.time() - t_start
            speed = (step + 1) / elapsed
            print(f"    Step [{step+1:4d}/{actual_steps}] | Loss: {loss.item():.4f} (L1={l1.item():.4f}, SSIM={1.0-l_ssim.item():.4f}) | Speed: {speed:.1f} steps/s")

    # Final PSNR on frame 0
    with torch.no_grad():
        ren0_final, _, _ = gsplat.rasterization(
            means=means,
            quats=F.normalize(quats, dim=-1),
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            viewmats=viewmats_cu[0:1],
            Ks=Ks_cu[0:1],
            width=W,
            height=H,
            sh_degree=None,
        )
        final_mse = F.mse_loss(ren0_final[0], gt_colors[0]).item()
        final_psnr = -10.0 * math.log10(max(final_mse, 1e-8))

    print(f"  [✓] Optimization done in {time.time()-t_start:.1f}s | Frame 0 PSNR: {init_psnr:.2f} dB -> {final_psnr:.2f} dB (+{final_psnr - init_psnr:.2f} dB)")

    # 5. Pruning floaters & outliers
    with torch.no_grad():
        opt_means = means.detach().cpu().numpy()
        opt_scales = torch.exp(log_scales).detach().cpu().numpy()
        opt_quats = F.normalize(quats, dim=-1).detach().cpu().numpy()
        opt_opacities = torch.sigmoid(opacity_logits).detach().cpu().numpy()
        opt_colors = torch.sigmoid(color_logits).detach().cpu().numpy()

        # Filter criteria: opacity >= prune_opacity and max scale <= 0.15m
        max_scale_dim = opt_scales.max(axis=-1)
        valid = (opt_opacities >= prune_opacity) & (max_scale_dim <= 0.15) & (max_scale_dim >= 0.0001)

        print(f"  Pruning: {N_gaussians:,} -> {valid.sum():,} Gaussians kept (removed {N_gaussians - valid.sum():,} low-confidence floaters)")

        res = {
            "means": opt_means[valid],
            "scales": opt_scales[valid],
            "quats": opt_quats[valid],
            "opacities": opt_opacities[valid],
            "colors": opt_colors[valid],
        }

        # Clean GPU memory
        del means, quats, log_scales, opacity_logits, color_logits, opt, ssim_window
        del ren0_init, ren0_final, gt_colors, gt_masks, Ks_cu, viewmats_cu
        torch.cuda.empty_cache()

        return res


def transform_gaussians_sim3(
    gaussians: Dict[str, np.ndarray],
    scale: float,
    transform_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Applies Sim(3) similarity transformation: y = s * R * x + t to 3D Gaussians.
    - Means: mu' = s * mu * R^T + t
    - Scales: S' = s * S
    - Quaternions: q' = q_R * q
    - Opacities & Colors: invariant
    """
    R_mat = transform_matrix[:3, :3]
    t_vec = transform_matrix[:3, 3]

    # Transform Means
    means_scaled = gaussians["means"] * scale
    means_trans = (means_scaled @ R_mat.T) + t_vec

    # Transform Scales
    scales_trans = gaussians["scales"] * scale

    # Transform Quaternions
    r_rot = R.from_matrix(R_mat)
    q_scipy = r_rot.as_quat()  # (x, y, z, w)
    q_R_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float32)
    quats_trans = quat_multiply(q_R_wxyz, gaussians["quats"])

    # Renormalize quaternions
    quats_trans /= (np.linalg.norm(quats_trans, axis=-1, keepdims=True) + 1e-8)

    return {
        "means": means_trans.astype(np.float32),
        "scales": scales_trans.astype(np.float32),
        "quats": quats_trans.astype(np.float32),
        "opacities": gaussians["opacities"].astype(np.float32),
        "colors": gaussians["colors"].astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Robot Photorealistic 3DGS Training Pipeline")
    parser.add_argument("--scene-dir", type=str, required=True, help="Path to scene directory containing transforms.json")
    parser.add_argument("--streams-dir", type=str, default="outputs/streams", help="Path to streams parent directory")
    parser.add_argument("--transforms-json", type=str, default=None, help="Optional explicit path to transforms.json")
    parser.add_argument("--steps-per-robot", type=int, default=800, help="Training steps per robot video stream")
    parser.add_argument("--output-ply", type=str, default=None, help="Output global 3DGS .ply path")
    parser.add_argument("--output-splat", type=str, default=None, help="Output global 3DGS .splat path")
    parser.add_argument("--save-subscenes", action="store_true", default=True, help="Save individual robot .splat/.ply into aligned_individual/")
    parser.add_argument("--invert-z", action="store_true", default=False, help="Invert Z axis on final export")
    parser.add_argument("--prune-opacity", type=float, default=0.05, help="Prune Gaussians with opacity below threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    streams_dir = Path(args.streams_dir)
    transforms_path = Path(args.transforms_json) if args.transforms_json else scene_dir / "transforms.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")

    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sequences_meta = meta.get("sequences", {})
    if not sequences_meta:
        raise ValueError(f"No sequences found in {transforms_path}")

    print(f"=== Starting Multi-Robot 3DGS Photometric Training ===")
    print(f"Scene Directory: {scene_dir}")
    print(f"Found {len(sequences_meta)} sequences in registration tree.")

    global_means_list = []
    global_scales_list = []
    global_quats_list = []
    global_opacities_list = []
    global_colors_list = []

    individual_dir = scene_dir / "aligned_individual"
    if args.save_subscenes:
        individual_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()

    for seq_name, sinfo in sequences_meta.items():
        seq_stream_dir = streams_dir / seq_name
        if not seq_stream_dir.exists():
            print(f"[-] Warning: stream dir {seq_stream_dir} does not exist, skipping.")
            continue

        scale = float(sinfo.get("scale_to_anchor", 1.0))
        T = np.array(sinfo.get("transform_matrix", np.eye(4)), dtype=np.float32)

        # 1. Train local Gaussians for this robot
        local_gaussians = train_single_sequence(
            seq_name=seq_name,
            stream_dir=seq_stream_dir,
            total_steps=args.steps_per_robot,
            prune_opacity=args.prune_opacity,
            device=args.device,
        )

        # 2. Transform into global anchor frame via Sim(3)
        print(f"  Transforming Gaussians to anchor frame (scale={scale:.4f})...")
        trans_gaussians = transform_gaussians_sim3(local_gaussians, scale, T)

        # 3. Optional Save Individual Subscene
        if args.save_subscenes:
            robot_id = seq_name.split("-")[-1]
            sub_splat = individual_dir / f"{robot_id}_3dgs.splat"
            sub_ply = individual_dir / f"{robot_id}_3dgs.ply"

            # Prepare types
            xyz = trans_gaussians["means"].copy()
            if args.invert_z:
                xyz[:, 2] = -xyz[:, 2]
            scales = trans_gaussians["scales"]
            quats = trans_gaussians["quats"]
            opacities = trans_gaussians["opacities"]
            colors_rgb = trans_gaussians["colors"]
            rgb_u8 = np.clip(np.round(colors_rgb * 255.0), 0, 255).astype(np.uint8)

            write_splat_binary(
                filepath=sub_splat,
                xyz=xyz,
                rgb_uint8=rgb_u8,
                opacity_alpha=opacities,
                scales_m=scales,
                rot_quats=quats,
            )

            # PLY log-scales & logit opacities
            log_scales = np.log(np.maximum(scales, 1e-6))
            opac_clamped = np.clip(opacities, 1e-4, 1.0 - 1e-4)
            opac_logits = np.log(opac_clamped / (1.0 - opac_clamped))
            f_dc = (colors_rgb - 0.5) / SH_C0
            normals = np.zeros_like(xyz)

            write_3dgs_ply(
                filepath=sub_ply,
                xyz=xyz,
                normals=normals,
                f_dc=f_dc,
                opacity_logit=opac_logits,
                log_scales=log_scales,
                rot_quats=quats,
                sh_degree=0,
            )
            print(f"  Saved individual sub-scene: {sub_splat.name} ({len(xyz):,} Gaussians)")

        global_means_list.append(trans_gaussians["means"])
        global_scales_list.append(trans_gaussians["scales"])
        global_quats_list.append(trans_gaussians["quats"])
        global_opacities_list.append(trans_gaussians["opacities"])
        global_colors_list.append(trans_gaussians["colors"])

    if not global_means_list:
        print("Error: No Gaussians were trained.")
        sys.exit(1)

    # 4. Merge All Sequences into Global Scene
    all_means = np.concatenate(global_means_list, axis=0)
    all_scales = np.concatenate(global_scales_list, axis=0)
    all_quats = np.concatenate(global_quats_list, axis=0)
    all_opacities = np.concatenate(global_opacities_list, axis=0)
    all_colors = np.concatenate(global_colors_list, axis=0)

    total_gaussians = len(all_means)
    print(f"\n=======================================================")
    print(f"[*] Global Multi-Robot Scene Assembled: {total_gaussians:,} Gaussians")
    print(f"=======================================================")

    # 5. Handle Invert Z if requested
    if args.invert_z:
        print("  Applying coordinate inversion: Z -> -Z...")
        all_means[:, 2] = -all_means[:, 2]

    # 6. Export Global Deliverables
    out_splat = Path(args.output_splat) if args.output_splat else scene_dir / "reconstruction_3dgs.splat"
    out_ply = Path(args.output_ply) if args.output_ply else scene_dir / "reconstruction_3dgs.ply"

    rgb_uint8 = np.clip(np.round(all_colors * 255.0), 0, 255).astype(np.uint8)

    print(f"Exporting compact .splat to: {out_splat}")
    splat_bytes = write_splat_binary(
        filepath=out_splat,
        xyz=all_means,
        rgb_uint8=rgb_uint8,
        opacity_alpha=all_opacities,
        scales_m=all_scales,
        rot_quats=all_quats,
    )

    print(f"Exporting 3DGS PLY to: {out_ply}")
    log_scales = np.log(np.maximum(all_scales, 1e-6))
    opac_clamped = np.clip(all_opacities, 1e-4, 1.0 - 1e-4)
    opac_logits = np.log(opac_clamped / (1.0 - opac_clamped))
    f_dc = (all_colors - 0.5) / SH_C0
    normals = np.zeros_like(all_means)

    ply_bytes = write_3dgs_ply(
        filepath=out_ply,
        xyz=all_means,
        normals=normals,
        f_dc=f_dc,
        opacity_logit=opac_logits,
        log_scales=log_scales,
        rot_quats=all_quats,
        sh_degree=0,
    )

    total_time = time.time() - t_total_start
    print(f"\n=======================================================")
    print(f"[✓] Pipeline Completed Successfully in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Total Photorealistic Gaussians: {total_gaussians:,}")
    print(f"  .splat size: {splat_bytes / 1024 / 1024:.2f} MB ({out_splat})")
    print(f"  .ply size:   {ply_bytes / 1024 / 1024:.2f} MB ({out_ply})")
    print(f"  WebGL Viewer URL: http://0.0.0.0:8088/splat?model={out_splat}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
