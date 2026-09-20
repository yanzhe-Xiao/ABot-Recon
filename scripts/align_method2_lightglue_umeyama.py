#!/usr/bin/env python3
"""
Method 2: Video-Assisted Multimodal Fast Registration
(LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp)
According to outputs/任务.md 方案二.

Pipeline:
  1. Keyframe Extraction & Retrieval:
     Sample keyframes from both sequences and use DINO-SALAD global descriptors to find top overlapping pairs.
  2. 2D Fast Feature Matching:
     ALIKED feature extraction + LightGlue dynamic early-stopping matching.
  3. 2D-to-3D Lifting:
     Unproject 2D matched keypoints into 3D world coordinates in Map A and Map B.
  4. Umeyama Sim(3) Closed-Form Pose:
     RANSAC + Umeyama SVD closed-form solver (<1ms) to estimate scale s, rotation R, translation t.
  5. small_gicp Fine Registration:
     Scale and transform source cloud, then apply parallel VGICP to eliminate fine seam errors.
  6. Map Fusion & Export:
     Fuse aligned source and target point clouds with voxel grid de-duplication.
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
from abot_recon.sparse_loop.retrieval import RetrievalConfig, compute_descriptors
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Method 2: Video-Assisted Multimodal Fast Registration (LightGlue + Umeyama + small_gicp)"
    )
    parser.add_argument(
        "--recon-dir-a",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/mine_VID20260903181931_loop"),
        help="Reconstruction directory for Sequence A",
    )
    parser.add_argument(
        "--recon-dir-b",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/mine_VID20260903182041_loop"),
        help="Reconstruction directory for Sequence B",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data/xyz/ABot-Recon/outputs/alignment"),
        help="Output directory for aligned point clouds and report",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=10,
        help="Stride for keyframe sampling (default: 10 frames)",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=5,
        help="Number of candidate keyframe pairs to match with LightGlue (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device for deep feature models",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=3000,
        help="Number of RANSAC iterations for Umeyama pose estimation",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=0.06,
        help="RANSAC inlier distance threshold in meters (default: 0.06m)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel downsample size for small_gicp evaluation (default: 0.02m)",
    )
    parser.add_argument(
        "--merge-voxel-size",
        type=float,
        default=0.015,
        help="Voxel size for fused point cloud de-duplication (default: 0.015m)",
    )
    return parser.parse_args()


def umeyama_svd(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama closed-form SVD algorithm for SE(3) / Sim(3).
    Finds s, R, t such that: dst ≈ s * R @ src + t
    """
    assert src.shape == dst.shape
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
    if estimate_scale:
        s = float(np.sum(D * np.diag(S)) / var_src)
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    iters: int = 3000,
    thresh: float = 0.06,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """RANSAC wrapper around closed-form Umeyama solver."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        return None
    best_inliers = np.empty(0, dtype=np.int64)
    best_model = None

    for _ in range(iters):
        idx = np.random.choice(n, 4, replace=False)
        try:
            s, R, t = umeyama_svd(src[idx], dst[idx], estimate_scale=estimate_scale)
            if estimate_scale and (s < min_scale or s > max_scale):
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
        if estimate_scale and (s < min_scale or s > max_scale):
            return None
        pred = s * (src[best_inliers] @ R.T) + t
        rmse = float(np.sqrt(np.mean(np.sum((dst[best_inliers] - pred)**2, axis=1))))
        return s, R, t, best_inliers, rmse
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("Method 2: Video-Assisted Multimodal Fast Registration")
    print("  (LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp)")
    print("=" * 70)
    print(f"Sequence A: {args.recon_dir_a}")
    print(f"Sequence B: {args.recon_dir_b}")
    print(f"Device:     {device}")

    total_start_time = time.time()

    # 1. Load Reconstruction Tensors
    print("\n[Step 1] Loading reconstruction tensors...")
    t0 = time.time()
    colors_a = torch.load(args.recon_dir_a / "colors.pt", map_location="cpu", weights_only=True)
    colors_b = torch.load(args.recon_dir_b / "colors.pt", map_location="cpu", weights_only=True)
    world_a = torch.load(args.recon_dir_a / "world_points.pt", map_location="cpu", weights_only=True)
    world_b = torch.load(args.recon_dir_b / "world_points.pt", map_location="cpu", weights_only=True)
    conf_a = torch.load(args.recon_dir_a / "confidence.pt", map_location="cpu", weights_only=True)
    conf_b = torch.load(args.recon_dir_b / "confidence.pt", map_location="cpu", weights_only=True)

    source_ply_path = args.recon_dir_a / "reconstruction.ply"
    target_ply_path = args.recon_dir_b / "reconstruction.ply"
    pcd_src = o3d.io.read_point_cloud(str(source_ply_path))
    pcd_tgt = o3d.io.read_point_cloud(str(target_ply_path))
    print(f"  Sequence A frames: {len(colors_a)}, Point cloud points: {len(pcd_src.points):,}")
    print(f"  Sequence B frames: {len(colors_b)}, Point cloud points: {len(pcd_tgt.points):,}")
    print(f"  Data loaded in {time.time() - t0:.2f}s")

    # 2. Keyframe Extraction & Retrieval
    print(f"\n[Step 2] Keyframe retrieval (sampling stride = {args.keyframe_stride})...")
    t0 = time.time()
    idx_a = np.arange(0, len(colors_a), args.keyframe_stride)
    idx_b = np.arange(0, len(colors_b), args.keyframe_stride)

    retrieval_cfg = RetrievalConfig(
        salad_checkpoint=Path("/home/data/xyz/ABot-Recon/checkpoints/loop/dino_salad.ckpt"),
        dino_checkpoint=Path("/home/data/xyz/ABot-Recon/checkpoints/loop/dinov2_vitb14_pretrain.pth"),
        backbone="dinov2_vitb14",
        verbose=False,
    )
    desc_a = compute_descriptors(colors_a[idx_a].numpy(), retrieval_cfg, device)
    desc_b = compute_descriptors(colors_b[idx_b].numpy(), retrieval_cfg, device)

    sim_matrix = desc_a @ desc_b.T
    retrieval_time = time.time() - t0

    top_flat_indices = np.argsort(-sim_matrix, axis=None)[: args.top_k_pairs * 2]
    candidate_pairs: list[tuple[int, int, float]] = []
    seen_a: set[int] = set()
    for flat_idx in top_flat_indices:
        i, j = np.unravel_index(flat_idx, sim_matrix.shape)
        fa, fb = int(idx_a[i]), int(idx_b[j])
        if fa not in seen_a and len(candidate_pairs) < args.top_k_pairs:
            candidate_pairs.append((fa, fb, float(sim_matrix[i, j])))
            seen_a.add(fa)

    print(f"  Retrieval completed in {retrieval_time * 1000:.1f}ms")
    print(f"  Top-{len(candidate_pairs)} candidate keyframe pairs:")
    for rank, (fa, fb, sim) in enumerate(candidate_pairs, 1):
        print(f"    Rank {rank}: Frame A={fa:03d} <---> Frame B={fb:03d} (sim = {sim:.4f})")

    # 3. ALIKED + LightGlue 2D Matching & 2D-to-3D Lifting
    print("\n[Step 3] ALIKED extraction + LightGlue matching + 2D-to-3D lifting...")
    t0 = time.time()
    extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    _ = time.time() - t0

    pair_results = []
    t_matching_start = time.time()

    for fa, fb, score in candidate_pairs:
        t_pair0 = time.time()
        img_a = colors_a[fa].permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        img_b = colors_b[fb].permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            feat_a = extractor.extract(img_a)
            feat_b = extractor.extract(img_b)
            match_out = matcher({"image0": feat_a, "image1": feat_b})
            feat_a, feat_b, match_out = [rbd(x) for x in [feat_a, feat_b, match_out]]

        matches = match_out["matches"]
        kpts_a = feat_a["keypoints"][matches[:, 0]].cpu().numpy()
        kpts_b = feat_b["keypoints"][matches[:, 1]].cpu().numpy()

        p3d_a, p3d_b = [], []
        for (xa, ya), (xb, yb) in zip(kpts_a, kpts_b):
            ia_y = min(max(int(round(ya)), 0), 279)
            ia_x = min(max(int(round(xa)), 0), 503)
            ib_y = min(max(int(round(yb)), 0), 279)
            ib_x = min(max(int(round(xb)), 0), 503)
            if conf_a[fa, ia_y, ia_x] > 0.05 and conf_b[fb, ib_y, ib_x] > 0.05:
                pa = world_a[fa, ia_y, ia_x].numpy()
                pb = world_b[fb, ib_y, ib_x].numpy()
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    p3d_a.append(pa)
                    p3d_b.append(pb)

        pair_time = time.time() - t_pair0
        p3d_a_arr = np.array(p3d_a)
        p3d_b_arr = np.array(p3d_b)

        if len(p3d_a_arr) >= 4:
            # Run Umeyama with RANSAC
            t_u0 = time.time()
            res = ransac_umeyama(
                p3d_a_arr,
                p3d_b_arr,
                estimate_scale=True,
                iters=args.ransac_iters,
                thresh=args.ransac_thresh,
            )
            umeyama_time = time.time() - t_u0

            if res is not None:
                s, R, t_vec, inliers, rmse = res
                inlier_ratio = len(inliers) / len(p3d_a_arr)
                pair_results.append({
                    "fa": fa,
                    "fb": fb,
                    "score": score,
                    "num_matches": len(matches),
                    "num_3d_pairs": len(p3d_a_arr),
                    "num_inliers": len(inliers),
                    "inlier_ratio": inlier_ratio,
                    "scale": s,
                    "R": R,
                    "t": t_vec,
                    "rmse": rmse,
                    "pair_time_ms": pair_time * 1000,
                    "umeyama_time_ms": umeyama_time * 1000,
                })
                print(
                    f"    Pair ({fa:03d}, {fb:03d}): {len(matches)} 2D matches -> "
                    f"{len(p3d_a_arr)} 3D pairs -> {len(inliers)} inliers ({inlier_ratio*100:.1f}%), "
                    f"scale s={s:.4f}, RMSE={rmse*1000:.1f}mm in {pair_time*1000:.1f}ms"
                )

    _ = time.time() - t_matching_start

    if not pair_results:
        raise RuntimeError("No valid candidate pair produced a reliable Umeyama solution!")

    # Pick the best pair by inlier ratio and RMSE
    pair_results.sort(key=lambda x: (-x["inlier_ratio"], x["rmse"]))
    best_pair = pair_results[0]
    best_scale = best_pair["scale"]
    best_R = best_pair["R"]
    best_t = best_pair["t"]

    print("\n[Step 4] Best Umeyama Sim(3) Pose Solution:")
    print(f"  Selected Pair: Frame A={best_pair['fa']:03d} <---> Frame B={best_pair['fb']:03d}")
    print(f"  Estimated Scale Factor s: {best_scale:.4f}")
    print("  Rotation R:\n", np.array2string(best_R, precision=4, suppress_small=True))
    print("  Translation t:\n", np.array2string(best_t, precision=4, suppress_small=True))
    print(f"  Inlier RMSE: {best_pair['rmse'] * 1000:.2f}mm ({best_pair['num_inliers']}/{best_pair['num_3d_pairs']} inliers)")
    print(f"  Umeyama closed-form solve time: {best_pair['umeyama_time_ms']:.2f}ms")

    # 4. small_gicp Fine Registration
    print("\n[Step 5] small_gicp fine registration (VGICP)...")
    t0 = time.time()
    da = pcd_src.voxel_down_sample(args.voxel_size)
    db = pcd_tgt.voxel_down_sample(args.voxel_size)

    # Apply Umeyama Sim(3): scale s, then R, t
    pts_src_down = np.asarray(da.points, dtype=np.float64) * best_scale
    pts_src_coarse = pts_src_down @ best_R.T + best_t

    pcd_coarse_eval = o3d.geometry.PointCloud()
    pcd_coarse_eval.points = o3d.utility.Vector3dVector(pts_src_coarse)

    eval_coarse_03 = o3d.pipelines.registration.evaluate_registration(
        pcd_coarse_eval, db, max_correspondence_distance=0.03
    )
    eval_coarse_05 = o3d.pipelines.registration.evaluate_registration(
        pcd_coarse_eval, db, max_correspondence_distance=0.05
    )
    print(f"  Umeyama coarse fitness (<3cm): {eval_coarse_03.fitness * 100:.2f}%, RMSE: {eval_coarse_03.inlier_rmse * 1000:.2f}mm")
    print(f"  Umeyama coarse fitness (<5cm): {eval_coarse_05.fitness * 100:.2f}%, RMSE: {eval_coarse_05.inlier_rmse * 1000:.2f}mm")

    # Refine with small_gicp VGICP
    gicp_result = small_gicp.align(
        target_points=np.asarray(db.points, dtype=np.float64),
        source_points=pts_src_coarse,
        init_T_target_source=np.eye(4, dtype=np.float64),
        registration_type="VGICP",
        voxel_resolution=args.voxel_size * 2.5,
        downsampling_resolution=args.voxel_size,
        max_correspondence_distance=args.voxel_size * 2.5,
        max_iterations=40,
        num_threads=8,
    )
    gicp_time = time.time() - t0
    T_gicp_delta = gicp_result.T_target_source

    # Compose final transformation: T_final = T_delta @ [R | t]
    T_fine_rigid = np.eye(4, dtype=np.float64)
    T_fine_rigid[:3, :3] = best_R
    T_fine_rigid[:3, 3] = best_t
    T_fine_total = T_gicp_delta @ T_fine_rigid

    print(f"  small_gicp completed in {gicp_time * 1000:.1f}ms")
    print(f"  Iterations: {gicp_result.iterations}, Inliers: {gicp_result.num_inliers}")

    # Evaluate Fine Registration
    pcd_fine_eval = o3d.geometry.PointCloud(pcd_coarse_eval).transform(T_gicp_delta)
    eval_fine_03 = o3d.pipelines.registration.evaluate_registration(
        pcd_fine_eval, db, max_correspondence_distance=0.03
    )
    eval_fine_05 = o3d.pipelines.registration.evaluate_registration(
        pcd_fine_eval, db, max_correspondence_distance=0.05
    )
    print(f"  Fine fitness (<3cm): {eval_fine_03.fitness * 100:.2f}%, RMSE: {eval_fine_03.inlier_rmse * 1000:.2f}mm")
    print(f"  Fine fitness (<5cm): {eval_fine_05.fitness * 100:.2f}%, RMSE: {eval_fine_05.inlier_rmse * 1000:.2f}mm")

    # 5. Map Fusion & Export
    print("\n[Step 6] Fusing point clouds and exporting deliverables...")
    t0 = time.time()
    pts_full_src = np.asarray(pcd_src.points, dtype=np.float64) * best_scale
    pts_full_aligned = (pts_full_src @ T_fine_total[:3, :3].T) + T_fine_total[:3, 3]

    pcd_src_aligned = o3d.geometry.PointCloud()
    pcd_src_aligned.points = o3d.utility.Vector3dVector(pts_full_aligned)
    pcd_src_aligned.colors = pcd_src.colors

    merged_raw = pcd_src_aligned + pcd_tgt
    print(f"  Merged raw points: {len(merged_raw.points):,}")

    merged_pcd = merged_raw.voxel_down_sample(voxel_size=args.merge_voxel_size)
    print(f"  Fused and de-duplicated points (voxel={args.merge_voxel_size}m): {len(merged_pcd.points):,}")
    fusion_time = time.time() - t0
    print(f"  Fusion completed in {fusion_time * 1000:.1f}ms")

    aligned_ply_path = args.output_dir / "method2_lightglue_umeyama_aligned.ply"
    merged_ply_path = args.output_dir / "method2_lightglue_umeyama_merged.ply"
    full_merged_ply_path = args.output_dir / "method2_lightglue_umeyama_full_merged.ply"
    transform_json_path = args.output_dir / "method2_lightglue_umeyama_transform.json"

    o3d.io.write_point_cloud(str(aligned_ply_path), pcd_src_aligned)
    o3d.io.write_point_cloud(str(merged_ply_path), merged_pcd)
    o3d.io.write_point_cloud(str(full_merged_ply_path), merged_raw)
    total_time = time.time() - total_start_time


    # Save summary metadata JSON
    result_data = {
        "method": "Method 2: LightGlue + 2D-to-3D Lifting + Umeyama + small_gicp",
        "type": "Video-Assisted Multimodal Sim(3) Registration",
        "source_ply": str(source_ply_path),
        "target_ply": str(target_ply_path),
        "selected_pair": {
            "frame_a": best_pair["fa"],
            "frame_b": best_pair["fb"],
            "num_2d_matches": best_pair["num_matches"],
            "num_3d_pairs": best_pair["num_3d_pairs"],
            "num_inliers": best_pair["num_inliers"],
            "inlier_ratio": round(best_pair["inlier_ratio"], 4),
        },
        "sim3_transformation": {
            "scale_s": round(float(best_scale), 6),
            "rotation_R": best_R.tolist(),
            "translation_t": best_t.tolist(),
            "umeyama_rmse_mm": round(float(best_pair["rmse"] * 1000), 2),
        },
        "fine_transformation_matrix": T_fine_total.tolist(),
        "timing_ms": {
            "keyframe_retrieval_ms": round(retrieval_time * 1000, 2),
            "lightglue_matching_ms": round(best_pair["pair_time_ms"], 2),
            "umeyama_solve_ms": round(best_pair["umeyama_time_ms"], 2),
            "small_gicp_fine_ms": round(gicp_time * 1000, 2),
            "fast_single_shot_pipeline_ms": round(
                (retrieval_time * 1000 + best_pair["pair_time_ms"] + best_pair["umeyama_time_ms"] + gicp_time * 1000), 2
            ),
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

    print(f"\n  Saved aligned source PLY: {aligned_ply_path}")
    print(f"  Saved merged PLY:         {merged_ply_path}")
    print(f"  Saved transform JSON:     {transform_json_path}")
    print("=" * 70)
    print(f"Method 2 Finished Successfully in {total_time:.2f}s!")
    print(f"Final Fine Fitness (<5cm): {eval_fine_05.fitness * 100:.2f}% | RMSE: {eval_fine_05.inlier_rmse * 1000:.2f}mm")
    print(f"Final Fine Fitness (<3cm): {eval_fine_03.fitness * 100:.2f}% | RMSE: {eval_fine_03.inlier_rmse * 1000:.2f}mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
