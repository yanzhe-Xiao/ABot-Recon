#!/usr/bin/env python3
"""
R3PM-Net Point Cloud Registration and Stitching Pipeline.

Uses R3PM-Net (Real-time, Robust, Real-world Point Matching Network)
to register and merge two 3D point clouds:
- Source: outputs/mine_VID20260903181931_loop/reconstruction.ply
- Target: outputs/mine_VID20260903182041_loop/reconstruction.ply

Outputs:
- outputs/alignment/merged.ply
- outputs/alignment/mine_r3pm_net_full_merged.ply
- outputs/alignment/mine_r3pm_net_5mm_merged.ply
- outputs/alignment/mine_r3pm_net_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# Ensure R3PM-Net is on sys.path
R3PM_NET_ROOT = Path("/home/data/xyz/R3PM-Net")
if str(R3PM_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(R3PM_NET_ROOT))

# Also ensure ABot-Recon root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from r3pm_net.model import R3PMNet
from r3pm_net.feature_extractor import feature_extractor
from scripts.export_reconstruction_ply import write_binary_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3PM-Net Point Cloud Registration and Stitching")
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "outputs/mine_VID20260903181931_loop/reconstruction.ply",
        help="Path to source point cloud PLY",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / "outputs/mine_VID20260903182041_loop/reconstruction.ply",
        help="Path to target point cloud PLY",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=R3PM_NET_ROOT / "checkpoints/clean-trained.pth",
        help="Path to R3PM-Net / RPMNet pretrained checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/alignment",
        help="Output directory for merged point clouds",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of keypoints sampled for R3PM-Net inference (default: 2048)",
    )
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Perform hybrid fine GICP refinement after R3PM-Net global registration",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel size for 5mm downsampled merged cloud (default: 0.005m)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device for inference",
    )
    return parser.parse_args()


def sample_keypoints_with_normals(
    pcd: o3d.geometry.PointCloud, num_points: int, voxel_size: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud and compute surface normals for PPFNet."""
    # Voxel downsample first to distribute points uniformly in 3D space
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(down.points) < num_points:
        down = pcd

    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )

    pts = np.asarray(down.points)
    normals = np.asarray(down.normals)

    # Random / uniform choice to exact num_points
    np.random.seed(42)
    if len(pts) >= num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
    else:
        idx = np.random.choice(len(pts), num_points, replace=True)

    return pts[idx], normals[idx]


def run_r3pm_net(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    checkpoint_path: Path,
    num_points: int = 2048,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run R3PM-Net model inference to compute rigid transformation (R, t)."""
    print(f"[R3PM-Net] Extracting {num_points} keypoints and surface normals...")
    pts_s, norm_s = sample_keypoints_with_normals(source_pcd, num_points, voxel_size=0.15)
    pts_t, norm_t = sample_keypoints_with_normals(target_pcd, num_points, voxel_size=0.20)

    # Normalize to unit sphere (shared scale to preserve rigid isometry)
    c_s = pts_s.mean(axis=0)
    c_t = pts_t.mean(axis=0)
    s_s = np.max(np.linalg.norm(pts_s - c_s, axis=1))
    s_t = np.max(np.linalg.norm(pts_t - c_t, axis=1))
    scale = max(s_s, s_t, 1e-6)

    pts_s_norm = (pts_s - c_s) / scale
    pts_t_norm = (pts_t - c_t) / scale

    # Build input tensor of shape [1, N, 6] (XYZ + Normal)
    data_src = np.concatenate([pts_s_norm, norm_s], axis=-1)[np.newaxis, ...]
    data_tgt = np.concatenate([pts_t_norm, norm_t], axis=-1)[np.newaxis, ...]

    dev = torch.device(device)
    tensor_src = torch.from_numpy(data_src).float().to(dev)
    tensor_tgt = torch.from_numpy(data_tgt).float().to(dev)

    print(f"[R3PM-Net] Loading model and pretrained weights from {checkpoint_path}...")
    model = R3PMNet(feature_model=feature_extractor).to(dev)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("[R3PM-Net] Running neural forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(tensor_tgt, tensor_src, max_iterations=2)
    inference_time = time.perf_counter() - t0

    R_norm = output["est_R"][0].cpu().numpy()
    t_norm = output["est_t"][0].cpu().numpy()

    # Reconstruct physical rigid transformation:
    # P_tgt_norm = R_norm * P_src_norm + t_norm
    # (P_tgt - c_t) / scale = R_norm * (P_src - c_s) / scale + t_norm
    # P_tgt = R_norm * P_src + (c_t - R_norm * c_s + scale * t_norm)
    R = R_norm
    t = c_t - R @ c_s + scale * t_norm
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    print(f"[R3PM-Net] Inference completed in {inference_time * 1000:.1f} ms.")
    return R, t, T, inference_time


def refine_with_gicp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    max_corr_dist: float = 0.30,
) -> tuple[np.ndarray, float, float]:
    """Refine coarse alignment using Generalized ICP (R3PM-Net coarse-to-fine pipeline)."""
    print("[Refinement] Running hybrid fine GICP alignment...")
    src_down = source_pcd.voxel_down_sample(0.05)
    tgt_down = target_pcd.voxel_down_sample(0.05)
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
    )
    method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(epsilon=0.001)

    result = o3d.pipelines.registration.registration_generalized_icp(
        src_down, tgt_down, max_correspondence_distance=max_corr_dist,
        init=init_T, estimation_method=method, criteria=criteria
    )
    print(f"[Refinement] GICP Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f} m")
    return result.transformation, result.fitness, result.inlier_rmse


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("R3PM-Net Point Cloud Registration and Stitching")
    print("=" * 60)
    print(f"Source file: {args.source}")
    print(f"Target file: {args.target}")
    print(f"Output dir:  {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Point Clouds
    print("\n[Step 1/5] Loading 3D point clouds...")
    t_start = time.perf_counter()
    pcd_src = o3d.io.read_point_cloud(str(args.source))
    pcd_tgt = o3d.io.read_point_cloud(str(args.target))

    pts_src = np.asarray(pcd_src.points)
    col_src = (np.asarray(pcd_src.colors) * 255.0).round().astype(np.uint8) if pcd_src.has_colors() else np.full((len(pts_src), 3), 180, dtype=np.uint8)

    pts_tgt = np.asarray(pcd_tgt.points)
    col_tgt = (np.asarray(pcd_tgt.colors) * 255.0).round().astype(np.uint8) if pcd_tgt.has_colors() else np.full((len(pts_tgt), 3), 180, dtype=np.uint8)

    print(f"Source points: {len(pts_src):,} | Target points: {len(pts_tgt):,}")

    # 2. Model Inference
    print("\n[Step 2/5] Running R3PM-Net global registration...")
    R_coarse, t_coarse, T_coarse, infer_time = run_r3pm_net(
        pcd_src, pcd_tgt, args.checkpoint, num_points=args.num_points, device=args.device
    )

    print("\nR3PM-Net Rotation Matrix R:")
    print(np.array2string(R_coarse, precision=6, suppress_small=True))
    print("\nR3PM-Net Translation Vector t:")
    print(np.array2string(t_coarse, precision=6, suppress_small=True))

    # 3. Optional Fine Refinement (Coarse-to-Fine)
    final_T = T_coarse
    fitness_val = None
    rmse_val = None
    if args.refine:
        print("\n[Step 3/5] Performing fine GICP refinement...")
        final_T, fitness_val, rmse_val = refine_with_gicp(pcd_src, pcd_tgt, T_coarse)
        R_final = final_T[:3, :3]
        t_final = final_T[:3, 3]
        print("\nFinal Refined Transformation Matrix T:")
        print(np.array2string(final_T, precision=6, suppress_small=True))
    else:
        R_final = R_coarse
        t_final = t_coarse

    # Evaluate Registration
    eval_05 = o3d.pipelines.registration.evaluate_registration(
        pcd_src, pcd_tgt, max_correspondence_distance=0.50, transformation=final_T
    )
    eval_02 = o3d.pipelines.registration.evaluate_registration(
        pcd_src, pcd_tgt, max_correspondence_distance=0.20, transformation=final_T
    )
    print("\nRegistration Evaluation Metrics:")
    print(f"  Max correspondence distance 0.50m: Fitness = {eval_05.fitness:.4f}, Inlier RMSE = {eval_05.inlier_rmse:.4f} m")
    print(f"  Max correspondence distance 0.20m: Fitness = {eval_02.fitness:.4f}, Inlier RMSE = {eval_02.inlier_rmse:.4f} m")

    # 4. Point Cloud Stitching
    print("\n[Step 4/5] Applying rigid transformation and stitching point clouds...")
    # Transform source points
    pts_src_trans = pts_src @ R_final.T + t_final

    # 100% full detail merged point cloud (zero point loss)
    merged_pts = np.vstack([pts_src_trans, pts_tgt])
    merged_col = np.vstack([col_src, col_tgt])
    print(f"Total merged points: {len(merged_pts):,} ({len(pts_src):,} src + {len(pts_tgt):,} tgt)")

    # 5. Result Saving
    print("\n[Step 5/5] Saving merged point cloud PLY files...")
    # Primary deliverable: merged.ply
    primary_ply = args.output_dir / "merged.ply"
    write_binary_ply(
        primary_ply,
        merged_pts.astype(np.float32),
        merged_col,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 3), dtype=np.uint8),
    )
    print(f"  -> Saved {primary_ply} ({primary_ply.stat().st_size / 1024 / 1024:.1f} MB, {len(merged_pts):,} points)")

    # Descriptive full merged PLY
    full_ply = args.output_dir / "mine_r3pm_net_full_merged.ply"
    write_binary_ply(
        full_ply,
        merged_pts.astype(np.float32),
        merged_col,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 3), dtype=np.uint8),
    )
    print(f"  -> Saved {full_ply} ({full_ply.stat().st_size / 1024 / 1024:.1f} MB)")

    # 5mm Voxel Grid Dedup Merged PLY
    print("\nGenerating 5mm voxel-grid filtered smooth merged PLY...")
    pcd_merged_full = o3d.geometry.PointCloud()
    pcd_merged_full.points = o3d.utility.Vector3dVector(merged_pts)
    pcd_merged_full.colors = o3d.utility.Vector3dVector(merged_col / 255.0)
    pcd_merged_5mm = pcd_merged_full.voxel_down_sample(args.voxel_size)

    pts_5mm = np.asarray(pcd_merged_5mm.points)
    col_5mm = (np.asarray(pcd_merged_5mm.colors) * 255.0).round().astype(np.uint8)

    merged_5mm_ply = args.output_dir / "mine_r3pm_net_5mm_merged.ply"
    write_binary_ply(
        merged_5mm_ply,
        pts_5mm.astype(np.float32),
        col_5mm,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 3), dtype=np.uint8),
    )
    print(f"  -> Saved {merged_5mm_ply} ({merged_5mm_ply.stat().st_size / 1024 / 1024:.1f} MB, {len(pts_5mm):,} points)")

    # Also save camera poses for trajectory visualization
    # Load source and target poses if available
    src_poses_file = args.source.parent / "camera_poses.npy"
    tgt_poses_file = args.target.parent / "camera_poses.npy"
    if src_poses_file.is_file() and tgt_poses_file.is_file():
        poses_src = np.load(src_poses_file)
        poses_tgt = np.load(tgt_poses_file)
        # Transform source camera poses: T_final @ pose_src
        poses_src_trans = np.matmul(final_T, poses_src)
        combined_poses = np.concatenate([poses_src_trans, poses_tgt], axis=0).astype(np.float32)
        np.save(args.output_dir / f"{merged_full_ply.stem}_poses.npy", combined_poses)
        np.save(args.output_dir / f"{merged_5mm_ply.stem}_poses.npy", combined_poses)
        np.save(args.output_dir / "mine_r3pm_net_poses.npy", combined_poses)
        print(f"  -> Saved {merged_full_ply.stem}_poses.npy ({len(combined_poses)} poses, float32)")

    # Save JSON Report
    total_time = time.perf_counter() - t_start
    results = {
        "method": "R3PM-Net (Real-time, Robust, Real-world Point Matching Network)",
        "source": str(args.source),
        "target": str(args.target),
        "source_points": len(pts_src),
        "target_points": len(pts_tgt),
        "merged_points_full": len(merged_pts),
        "merged_points_5mm": len(pts_5mm),
        "r3pm_net_inference_time_ms": round(infer_time * 1000, 2),
        "total_processing_time_sec": round(total_time, 2),
        "T_coarse_r3pm": T_coarse.tolist(),
        "R_coarse_r3pm": R_coarse.tolist(),
        "t_coarse_r3pm": t_coarse.tolist(),
        "T_final": final_T.tolist(),
        "R_final": R_final.tolist(),
        "t_final": t_final.tolist(),
        "metrics_05m": {
            "fitness": round(eval_05.fitness, 4),
            "inlier_rmse": round(eval_05.inlier_rmse, 4),
        },
        "metrics_02m": {
            "fitness": round(eval_02.fitness, 4),
            "inlier_rmse": round(eval_02.inlier_rmse, 4),
        },
    }

    report_path = args.output_dir / "mine_r3pm_net_results.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved results summary to {report_path}")

    print("\n" + "=" * 60)
    print("R3PM-Net Point Cloud Registration and Stitching Finished Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
