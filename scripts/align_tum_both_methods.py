#!/usr/bin/env python3
"""
TUM 360 Loop + TUM Desk Loop Alignment and High-Detail Fusion
Implements both Method 1 and Method 2 on TUM RGB-D dataset:
  - Method 1: Pure 3D Geometric (KISS-Matcher + small_gicp, SE3)
  - Method 2: Multimodal Video-Assisted (LightGlue + 2D-3D + Umeyama, Sim3)
Generates high-detail fused point clouds (~2.5 million points at 5mm resolution).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import small_gicp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_reconstruction_ply import write_binary_ply
import kiss_matcher
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TUM Dual-Method High-Detail Alignment and Fusion")
    parser.add_argument(
        "--tum-360-dir",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/tum_360_loop"),
        help="Directory containing tum_360_loop outputs",
    )
    parser.add_argument(
        "--tum-desk-dir",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/tum_desk_loop"),
        help="Directory containing tum_desk_loop outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/alignment"),
        help="Output directory for merged point clouds",
    )
    parser.add_argument(
        "--merge-voxel-size",
        type=float,
        default=0.005,
        help="Voxel size for fused point cloud detail retention (default: 0.005m = 5mm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch compute device",
    )
    return parser.parse_args()


def umeyama_svd(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    n, m = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.mean(np.sum(src_c**2, axis=1))
    if var_src < 1e-12:
        return 1.0, np.eye(3), np.zeros(3)
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float(np.sum(D * np.diag(S)) / var_src) if estimate_scale else 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    iters: int = 3000,
    thresh: float = 0.06,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None:
    n = len(src)
    if n < 4:
        return None
    best_inliers = np.empty(0, dtype=np.int64)
    best_model = None
    for _ in range(iters):
        idx = np.random.choice(n, 4, replace=False)
        try:
            s, R, t = umeyama_svd(src[idx], dst[idx], estimate_scale=estimate_scale)
            if estimate_scale and (s < 0.3 or s > 3.0):
                continue
            pred = s * (src @ R.T) + t
            err = np.linalg.norm(dst - pred, axis=1)
            inliers = np.where(err < thresh)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (s, R, t)
        except Exception:
            continue
    if len(best_inliers) >= 4 and best_model is not None:
        s, R, t = umeyama_svd(src[best_inliers], dst[best_inliers], estimate_scale=estimate_scale)
        pred = s * (src[best_inliers] @ R.T) + t
        rmse = float(np.sqrt(np.mean(np.sum((dst[best_inliers] - pred)**2, axis=1))))
        return s, R, t, best_inliers, rmse
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 75)
    print("TUM RGB-D Benchmark: tum_360_loop + tum_desk_loop Alignment & Fusion")
    print("=" * 75)

    # 1. Load Raw Data
    print("\n[Step 1] Loading raw TUM point clouds and reconstruction tensors...")
    pcd_360 = o3d.io.read_point_cloud(str(args.tum_360_dir / "reconstruction.ply"))
    pcd_desk = o3d.io.read_point_cloud(str(args.tum_desk_dir / "reconstruction.ply"))

    colors_desk = torch.load(args.tum_desk_dir / "colors.pt", map_location="cpu", weights_only=True)
    colors_360 = torch.load(args.tum_360_dir / "colors.pt", map_location="cpu", weights_only=True)
    world_desk = torch.load(args.tum_desk_dir / "world_points.pt", map_location="cpu", weights_only=True)
    world_360 = torch.load(args.tum_360_dir / "world_points.pt", map_location="cpu", weights_only=True)
    conf_desk = torch.load(args.tum_desk_dir / "confidence.pt", map_location="cpu", weights_only=True)
    conf_360 = torch.load(args.tum_360_dir / "confidence.pt", map_location="cpu", weights_only=True)

    print(f"  Target tum_360 raw points:  {len(pcd_360.points):,}")
    print(f"  Source tum_desk raw points: {len(pcd_desk.points):,}")
    print(f"  Total raw points combined:  {len(pcd_360.points) + len(pcd_desk.points):,}")

    # Downsampled for coarse alignment
    eval_vs = 0.03
    d_360 = pcd_360.voxel_down_sample(eval_vs)
    d_desk = pcd_desk.voxel_down_sample(eval_vs)
    pts_360 = np.asarray(d_360.points, dtype=np.float32)
    pts_desk = np.asarray(d_desk.points, dtype=np.float32)

    # =========================================================================
    # Method 1: Pure 3D Geometric Registration (KISS-Matcher + small_gicp)
    # =========================================================================
    print("\n" + "-" * 75)
    print("Executing Method 1: Pure 3D Geometric (KISS-Matcher + small_gicp)...")
    print("-" * 75)
    t0 = time.time()
    cfg = kiss_matcher.KISSMatcherConfig()
    cfg.voxel_size = eval_vs
    cfg.normal_radius = eval_vs * 3.0
    cfg.fpfh_radius = eval_vs * 5.0
    matcher_m1 = kiss_matcher.KISSMatcher(cfg)
    sol_m1 = matcher_m1.estimate(pts_desk, pts_360)
    m1_coarse_time = time.time() - t0

    T_m1_coarse = np.eye(4, dtype=np.float64)
    T_m1_coarse[:3, :3] = sol_m1.rotation
    T_m1_coarse[:3, 3] = sol_m1.translation
    print(f"  [Method 1] KISS-Matcher in {m1_coarse_time * 1000:.1f}ms: inliers={matcher_m1.get_num_final_inliers()}")

    t1 = time.time()
    res_m1_gicp = small_gicp.align(
        target_points=pts_360.astype(np.float64),
        source_points=pts_desk.astype(np.float64),
        init_T_target_source=T_m1_coarse,
        registration_type="PLANE_ICP",
        voxel_resolution=0.06,
        downsampling_resolution=eval_vs,
        max_correspondence_distance=0.06,
        max_iterations=60,
        num_threads=8,
    )
    m1_fine_time = time.time() - t1
    T_m1_fine = res_m1_gicp.T_target_source
    print(f"  [Method 1] small_gicp in {m1_fine_time * 1000:.1f}ms: inliers={res_m1_gicp.num_inliers}")

    pcd_m1_eval = o3d.geometry.PointCloud(d_desk).transform(T_m1_fine)
    ev_m1_5cm = o3d.pipelines.registration.evaluate_registration(pcd_m1_eval, d_360, 0.05)
    ev_m1_3cm = o3d.pipelines.registration.evaluate_registration(pcd_m1_eval, d_360, 0.03)
    print(f"  [Method 1 Results] Fitness @ 5cm: {ev_m1_5cm.fitness*100:.2f}%, RMSE: {ev_m1_5cm.inlier_rmse*1000:.2f}mm")
    print(f"  [Method 1 Results] Fitness @ 3cm: {ev_m1_3cm.fitness*100:.2f}%, RMSE: {ev_m1_3cm.inlier_rmse*1000:.2f}mm")

    # =========================================================================
    # Method 2: Video-Assisted Multimodal (LightGlue + Umeyama + small_gicp)
    # =========================================================================
    print("\n" + "-" * 75)
    print("Executing Method 2: Video-Assisted (LightGlue + Umeyama + small_gicp)...")
    print("-" * 75)
    t0 = time.time()
    extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    # Keyframe candidate pair (desk frame 50, 360 loop frame 0)
    fd, f360 = 50, 0
    img_d = colors_desk[fd].permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img_360 = colors_360[f360].permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        feat_d = extractor.extract(img_d)
        feat_360 = extractor.extract(img_360)
        match_out = matcher({"image0": feat_d, "image1": feat_360})
        feat_d, feat_360, match_out = [rbd(x) for x in [feat_d, feat_360, match_out]]

    matches = match_out["matches"]
    kpts_d = feat_d["keypoints"][matches[:, 0]].cpu().numpy()
    kpts_360 = feat_360["keypoints"][matches[:, 1]].cpu().numpy()

    p3d_d, p3d_360 = [], []
    for (xd, yd), (x360, y360) in zip(kpts_d, kpts_360):
        id_y, id_x = min(max(int(round(yd)), 0), 279), min(max(int(round(xd)), 0), 503)
        i360_y, i360_x = min(max(int(round(y360)), 0), 279), min(max(int(round(x360)), 0), 503)
        if conf_desk[fd, id_y, id_x] > 0.05 and conf_360[f360, i360_y, i360_x] > 0.05:
            p3d_d.append(world_desk[fd, id_y, id_x].numpy())
            p3d_360.append(world_360[f360, i360_y, i360_x].numpy())

    p3d_d = np.array(p3d_d)
    p3d_360 = np.array(p3d_360)
    m2_matching_time = time.time() - t0

    t_u0 = time.time()
    res_m2_u = ransac_umeyama(p3d_d, p3d_360, estimate_scale=True, thresh=0.06)
    m2_umeyama_time = time.time() - t_u0

    if res_m2_u is None:
        raise RuntimeError("Method 2 Umeyama solver failed to converge on TUM keyframes!")

    s_m2, R_m2, t_m2, inliers_m2, rmse_m2 = res_m2_u
    print(f"  [Method 2] LightGlue: {len(matches)} 2D matches -> {len(inliers_m2)}/{len(p3d_d)} 3D inliers in {m2_matching_time*1000:.1f}ms")
    print(f"  [Method 2] Umeyama: scale s={s_m2:.4f}, inlier RMSE={rmse_m2*1000:.2f}mm in {m2_umeyama_time*1000:.2f}ms")

    t_g0 = time.time()
    pts_d_scaled = np.asarray(d_desk.points, dtype=np.float64) * s_m2
    pts_d_coarse = pts_d_scaled @ R_m2.T + t_m2

    res_m2_gicp = small_gicp.align(
        target_points=pts_360.astype(np.float64),
        source_points=pts_d_coarse,
        init_T_target_source=np.eye(4),
        registration_type="PLANE_ICP",
        voxel_resolution=0.06,
        downsampling_resolution=eval_vs,
        max_correspondence_distance=0.06,
        max_iterations=60,
        num_threads=8,
    )
    _ = time.time() - t_g0

    T_m2_delta = res_m2_gicp.T_target_source
    T_m2_rigid = np.eye(4, dtype=np.float64)
    T_m2_rigid[:3, :3] = R_m2
    T_m2_rigid[:3, 3] = t_m2
    T_m2_fine_total = T_m2_delta @ T_m2_rigid

    pcd_m2_eval = o3d.geometry.PointCloud()
    pcd_m2_eval.points = o3d.utility.Vector3dVector(pts_d_coarse)
    pcd_m2_eval.transform(T_m2_delta)

    ev_m2_5cm = o3d.pipelines.registration.evaluate_registration(pcd_m2_eval, d_360, 0.05)
    ev_m2_3cm = o3d.pipelines.registration.evaluate_registration(pcd_m2_eval, d_360, 0.03)
    print(f"  [Method 2 Results] Fitness @ 5cm: {ev_m2_5cm.fitness*100:.2f}%, RMSE: {ev_m2_5cm.inlier_rmse*1000:.2f}mm")
    print(f"  [Method 2 Results] Fitness @ 3cm: {ev_m2_3cm.fitness*100:.2f}%, RMSE: {ev_m2_3cm.inlier_rmse*1000:.2f}mm")

    # =========================================================================
    # High-Detail Map Fusion (Voxel = 5mm)
    # =========================================================================
    fine_voxel = args.merge_voxel_size
    print("\n" + "-" * 75)
    print(f"Generating High-Detail Fused Point Clouds (Voxel = {fine_voxel * 1000:.1f} mm)...")
    print("-" * 75)

    # Fuse Method 1
    t0 = time.time()
    pcd_desk_m1_aligned = o3d.geometry.PointCloud(pcd_desk).transform(T_m1_fine)
    merged_m1_raw = pcd_desk_m1_aligned + pcd_360
    merged_m1_fine = merged_m1_raw.voxel_down_sample(fine_voxel)

    out_m1_ply = args.output_dir / "tum_method1_kiss_gicp_merged.ply"
    pts_m1_out = np.asarray(merged_m1_fine.points, dtype=np.float32)
    col_m1_out = (np.asarray(merged_m1_fine.colors) * 255.0).round().clip(0, 255).astype(np.uint8)
    write_binary_ply(out_m1_ply, pts_m1_out, col_m1_out, np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.uint8))
    print(f"  [Method 1] Saved high-detail fused cloud: {out_m1_ply.name} ({len(pts_m1_out):,} points, {out_m1_ply.stat().st_size / 1024 / 1024:.2f} MB)")

    # Fuse Method 2
    pts_desk_m2_full = np.asarray(pcd_desk.points, dtype=np.float64) * s_m2
    pts_desk_m2_aligned = (pts_desk_m2_full @ T_m2_fine_total[:3, :3].T) + T_m2_fine_total[:3, 3]
    pcd_desk_m2_aligned = o3d.geometry.PointCloud()
    pcd_desk_m2_aligned.points = o3d.utility.Vector3dVector(pts_desk_m2_aligned)
    pcd_desk_m2_aligned.colors = pcd_desk.colors

    merged_m2_raw = pcd_desk_m2_aligned + pcd_360
    merged_m2_fine = merged_m2_raw.voxel_down_sample(fine_voxel)

    out_m2_ply = args.output_dir / "tum_method2_lightglue_umeyama_merged.ply"
    pts_m2_out = np.asarray(merged_m2_fine.points, dtype=np.float32)
    col_m2_out = (np.asarray(merged_m2_fine.colors) * 255.0).round().clip(0, 255).astype(np.uint8)
    write_binary_ply(out_m2_ply, pts_m2_out, col_m2_out, np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.uint8))
    print(f"  [Method 2] Saved high-detail fused cloud: {out_m2_ply.name} ({len(pts_m2_out):,} points, {out_m2_ply.stat().st_size / 1024 / 1024:.2f} MB)")

    # Save summary metadata
    summary = {
        "dataset": "TUM RGB-D Benchmark (freiburg1_360 + freiburg1_desk)",
        "fusion_voxel_size_mm": fine_voxel * 1000,
        "method1": {
            "name": "KISS-Matcher + small_gicp",
            "type": "SE(3) Rigid",
            "scale": 1.0,
            "points_count": len(pts_m1_out),
            "file": str(out_m1_ply),
            "fitness_5cm": round(float(ev_m1_5cm.fitness * 100), 2),
            "rmse_5cm_mm": round(float(ev_m1_5cm.inlier_rmse * 1000), 2),
            "fitness_3cm": round(float(ev_m1_3cm.fitness * 100), 2),
            "rmse_3cm_mm": round(float(ev_m1_3cm.inlier_rmse * 1000), 2),
            "transformation": T_m1_fine.tolist(),
        },
        "method2": {
            "name": "LightGlue + Umeyama + small_gicp",
            "type": "Sim(3) Similarity",
            "scale": round(float(s_m2), 4),
            "points_count": len(pts_m2_out),
            "file": str(out_m2_ply),
            "fitness_5cm": round(float(ev_m2_5cm.fitness * 100), 2),
            "rmse_5cm_mm": round(float(ev_m2_5cm.inlier_rmse * 1000), 2),
            "fitness_3cm": round(float(ev_m2_3cm.fitness * 100), 2),
            "rmse_3cm_mm": round(float(ev_m2_3cm.inlier_rmse * 1000), 2),
            "transformation": T_m2_fine_total.tolist(),
        },
    }

    summary_path = args.output_dir / "tum_alignment_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metadata JSON: {summary_path}")
    print("=" * 75)
    print("TUM Dual-Method High-Detail Fusion Complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
