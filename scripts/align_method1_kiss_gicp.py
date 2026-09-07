#!/usr/bin/env python3
"""
Method 1: Pure 3D Geometric Registration (KISS-Matcher + small_gicp)
According to outputs/任务.md 方案一.

Pipeline:
  1. Preprocessing: Voxel downsample + normal estimation
  2. KISS-Matcher: Faster-PFH feature extraction + k-Core pruning + GNC solver -> T_coarse
  3. small_gicp: Multi-threaded parallel VGICP / GICP refinement -> T_fine
  4. Point Cloud Fusion: Transform source cloud and merge with target using voxel de-duplication
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import kiss_matcher
import numpy as np
import open3d as o3d
import small_gicp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Method 1: Pure 3D Geometric Registration (KISS-Matcher + small_gicp)"
    )
    parser.add_argument(
        "--source-ply",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/mine_VID20260903181931_loop/reconstruction.ply"),
        help="Path to source point cloud PLY",
    )
    parser.add_argument(
        "--target-ply",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/mine_VID20260903182041_loop/reconstruction.ply"),
        help="Path to target point cloud PLY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/alignment"),
        help="Output directory for aligned and merged point clouds",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="Voxel downsample size for coarse registration (default: 0.03m)",
    )
    parser.add_argument(
        "--merge-voxel-size",
        type=float,
        default=0.015,
        help="Voxel size for fused point cloud de-duplication (default: 0.015m)",
    )
    parser.add_argument(
        "--registration-type",
        choices=("VGICP", "GICP", "PLANE_ICP"),
        default="VGICP",
        help="Registration type for small_gicp (default: VGICP)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Thread count for small_gicp parallel execution",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=40,
        help="Max iterations for small_gicp optimization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Method 1: Pure 3D Geometric Registration (KISS-Matcher + small_gicp)")
    print("=" * 70)
    print(f"Source PLY: {args.source_ply}")
    print(f"Target PLY: {args.target_ply}")
    print(f"Output Dir: {args.output_dir}")

    total_start_time = time.time()

    # 1. Load Point Clouds
    print("\n[Step 1] Loading point clouds...")
    t0 = time.time()
    pcd_src = o3d.io.read_point_cloud(str(args.source_ply))
    pcd_tgt = o3d.io.read_point_cloud(str(args.target_ply))
    load_time = time.time() - t0
    print(f"  Source raw points: {len(pcd_src.points):,}")
    print(f"  Target raw points: {len(pcd_tgt.points):,}")
    print(f"  Loaded in {load_time:.2f}s")

    # Preprocessing: Voxel downsample
    print(f"\n[Step 2] Preprocessing: voxel downsampling (voxel_size = {args.voxel_size}m)...")
    t0 = time.time()
    src_down = pcd_src.voxel_down_sample(args.voxel_size)
    tgt_down = pcd_tgt.voxel_down_sample(args.voxel_size)

    # Estimate normals for robust ICP
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 3.0, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 3.0, max_nn=30)
    )

    pts_src = np.asarray(src_down.points, dtype=np.float32)
    pts_tgt = np.asarray(tgt_down.points, dtype=np.float32)
    prep_time = time.time() - t0
    print(f"  Source downsampled: {len(pts_src):,} points")
    print(f"  Target downsampled: {len(pts_tgt):,} points")
    print(f"  Preprocessing done in {prep_time * 1000:.1f}ms")

    # 2. KISS-Matcher Global Coarse Registration
    print("\n[Step 3] KISS-Matcher global coarse registration...")
    cfg = kiss_matcher.KISSMatcherConfig()
    cfg.voxel_size = args.voxel_size
    cfg.normal_radius = args.voxel_size * 3.0
    cfg.fpfh_radius = args.voxel_size * 5.0

    t0 = time.time()
    matcher = kiss_matcher.KISSMatcher(cfg)
    sol = matcher.estimate(pts_src, pts_tgt)
    coarse_time = time.time() - t0

    if not sol.valid:
        raise RuntimeError("KISS-Matcher failed to find a valid coarse registration solution!")

    T_coarse = np.eye(4, dtype=np.float64)
    T_coarse[:3, :3] = sol.rotation
    T_coarse[:3, 3] = sol.translation

    num_rot_inliers = matcher.get_num_rotation_inliers()
    num_final_inliers = matcher.get_num_final_inliers()
    print(f"  KISS-Matcher completed in {coarse_time * 1000:.1f}ms")
    print(f"  Rotation inliers: {num_rot_inliers}, Final inliers: {num_final_inliers}")
    print("  T_coarse:\n", np.array2string(T_coarse, precision=4, suppress_small=True))

    # Evaluate Coarse Registration
    pcd_coarse = o3d.geometry.PointCloud(src_down).transform(T_coarse)
    eval_coarse_03 = o3d.pipelines.registration.evaluate_registration(
        pcd_coarse, tgt_down, max_correspondence_distance=0.03
    )
    eval_coarse_05 = o3d.pipelines.registration.evaluate_registration(
        pcd_coarse, tgt_down, max_correspondence_distance=0.05
    )
    print(f"  Coarse fitness (<3cm): {eval_coarse_03.fitness * 100:.2f}%, RMSE: {eval_coarse_03.inlier_rmse * 1000:.2f}mm")
    print(f"  Coarse fitness (<5cm): {eval_coarse_05.fitness * 100:.2f}%, RMSE: {eval_coarse_05.inlier_rmse * 1000:.2f}mm")

    # 3. small_gicp Fast Fine Registration
    print(f"\n[Step 4] small_gicp fine registration ({args.registration_type})...")
    t0 = time.time()
    gicp_result = small_gicp.align(
        target_points=pts_tgt.astype(np.float64),
        source_points=pts_src.astype(np.float64),
        init_T_target_source=T_coarse,
        registration_type=args.registration_type,
        voxel_resolution=args.voxel_size * 3.0,
        downsampling_resolution=args.voxel_size,
        max_correspondence_distance=args.voxel_size * 2.5,
        max_iterations=args.max_iterations,
        num_threads=args.num_threads,
    )
    fine_time = time.time() - t0

    T_fine = gicp_result.T_target_source
    print(f"  small_gicp completed in {fine_time * 1000:.1f}ms")
    print(f"  Iterations: {gicp_result.iterations}, Inliers: {gicp_result.num_inliers}")
    print("  T_fine:\n", np.array2string(T_fine, precision=4, suppress_small=True))

    # Evaluate Fine Registration
    pcd_fine = o3d.geometry.PointCloud(src_down).transform(T_fine)
    eval_fine_03 = o3d.pipelines.registration.evaluate_registration(
        pcd_fine, tgt_down, max_correspondence_distance=0.03
    )
    eval_fine_05 = o3d.pipelines.registration.evaluate_registration(
        pcd_fine, tgt_down, max_correspondence_distance=0.05
    )
    print(f"  Fine fitness (<3cm): {eval_fine_03.fitness * 100:.2f}%, RMSE: {eval_fine_03.inlier_rmse * 1000:.2f}mm")
    print(f"  Fine fitness (<5cm): {eval_fine_05.fitness * 100:.2f}%, RMSE: {eval_fine_05.inlier_rmse * 1000:.2f}mm")

    # 4. Map Fusion & Export
    print("\n[Step 5] Applying transformation and fusing point clouds...")
    t0 = time.time()
    pcd_src_aligned = o3d.geometry.PointCloud(pcd_src)
    pcd_src_aligned.transform(T_fine)

    merged_raw = pcd_src_aligned + pcd_tgt
    print(f"  Merged raw points: {len(merged_raw.points):,}")

    merged_pcd = merged_raw.voxel_down_sample(voxel_size=args.merge_voxel_size)
    print(f"  Fused and de-duplicated points (voxel={args.merge_voxel_size}m): {len(merged_pcd.points):,}")
    fusion_time = time.time() - t0
    print(f"  Fusion done in {fusion_time * 1000:.1f}ms")

    # Save outputs
    aligned_ply_path = args.output_dir / "method1_kiss_gicp_aligned.ply"
    merged_ply_path = args.output_dir / "method1_kiss_gicp_merged.ply"
    full_merged_ply_path = args.output_dir / "method1_kiss_gicp_full_merged.ply"
    transform_json_path = args.output_dir / "method1_kiss_gicp_transform.json"

    print(f"\n[Step 6] Saving outputs to {args.output_dir}...")
    o3d.io.write_point_cloud(str(aligned_ply_path), pcd_src_aligned)
    o3d.io.write_point_cloud(str(merged_ply_path), merged_pcd)
    o3d.io.write_point_cloud(str(full_merged_ply_path), merged_raw)
    total_time = time.time() - total_start_time

    # Save summary metadata JSON
    result_data = {
        "method": "Method 1: KISS-Matcher + small_gicp",
        "type": "Pure 3D Geometric Registration (SE3)",
        "source_ply": str(args.source_ply),
        "target_ply": str(args.target_ply),
        "voxel_size": args.voxel_size,
        "registration_type": args.registration_type,
        "transformation_coarse": T_coarse.tolist(),
        "transformation_fine": T_fine.tolist(),
        "scale": 1.0,
        "timing_ms": {
            "preprocessing_ms": round(prep_time * 1000, 2),
            "kiss_matcher_coarse_ms": round(coarse_time * 1000, 2),
            "small_gicp_fine_ms": round(fine_time * 1000, 2),
            "total_registration_ms": round((prep_time + coarse_time + fine_time) * 1000, 2),
            "total_wall_time_s": round(total_time, 2),
        },
        "metrics": {
            "coarse_fitness_3cm": round(float(eval_coarse_03.fitness), 4),
            "coarse_rmse_3cm_mm": round(float(eval_coarse_03.inlier_rmse * 1000), 2),
            "coarse_fitness_5cm": round(float(eval_coarse_05.fitness), 4),
            "coarse_rmse_5cm_mm": round(float(eval_coarse_05.inlier_rmse * 1000), 2),
            "fine_fitness_3cm": round(float(eval_fine_03.fitness), 4),
            "fine_rmse_3cm_mm": round(float(eval_fine_03.inlier_rmse * 1000), 2),
            "fine_fitness_5cm": round(float(eval_fine_05.fitness), 4),
            "fine_rmse_5cm_mm": round(float(eval_fine_05.inlier_rmse * 1000), 2),
            "kiss_rotation_inliers": int(num_rot_inliers),
            "kiss_final_inliers": int(num_final_inliers),
            "small_gicp_inliers": int(gicp_result.num_inliers),
        },
        "point_counts": {
            "source_raw": len(pcd_src.points),
            "target_raw": len(pcd_tgt.points),
            "merged_fused": len(merged_pcd.points),
        },
        "saved_files": {
            "aligned_ply": str(aligned_ply_path),
            "merged_ply": str(merged_ply_path),
            "transform_json": str(transform_json_path),
        },
    }

    with open(transform_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved aligned source PLY: {aligned_ply_path}")
    print(f"  Saved merged PLY:         {merged_ply_path}")
    print(f"  Saved transform JSON:     {transform_json_path}")
    print("=" * 70)
    print(f"Method 1 Finished Successfully in {total_time:.2f}s!")
    print(f"Final Fine Fitness (<5cm): {eval_fine_05.fitness * 100:.2f}% | RMSE: {eval_fine_05.inlier_rmse * 1000:.2f}mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
