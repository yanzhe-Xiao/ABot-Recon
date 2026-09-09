#!/usr/bin/env python3
"""
Method 2: Video-Assisted Multimodal Fast Registration for 05, 06, 07, 08
(LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp VGICP)
Fuses 4 video streams into a unified 3D coordinate system.

Produces two versions:
  1. Normal true-color fusion (original RGB)
  2. Distinctly-colored fusion (each video stream assigned a unique color:
     05: Red, 06: Green, 07: Blue, 08: Gold/Yellow)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import small_gicp
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from abot_recon.sparse_loop.retrieval import RetrievalConfig, compute_descriptors
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
from scripts.align_method2_lightglue_umeyama import ransac_umeyama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Method 2: Multi-video point cloud fusion for 05-08 using LightGlue + Umeyama + small_gicp"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/alignment",
        help="Directory to save aligned deliverables",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=5,
        help="Stride for keyframe retrieval sampling",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=8,
        help="Top candidate keyframe pairs to test with LightGlue",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for ALIKED and LightGlue models",
    )
    parser.add_argument(
        "--merge-voxel-size",
        type=float,
        default=0.015,
        help="Voxel size for de-duplicated fused point cloud (default: 1.5cm)",
    )
    return parser.parse_args()


class MultimodalMatcher:
    def __init__(self, device: torch.device):
        self.device = device
        print(f"[Init] Loading ALIKED feature extractor and LightGlue matcher on {device}...")
        self.extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.retrieval_cfg = RetrievalConfig(
            salad_checkpoint=REPO_ROOT / "checkpoints/loop/dino_salad.ckpt",
            dino_checkpoint=REPO_ROOT / "checkpoints/loop/dinov2_vitb14_pretrain.pth",
            backbone="dinov2_vitb14",
            verbose=False,
        )

    def match_sequences(
        self,
        src_id: str,
        tgt_id: str,
        stride: int = 5,
        top_k: int = 8,
        ransac_thresh: float = 0.08,
    ) -> dict:
        t0 = time.time()
        dir_s = REPO_ROOT / f"outputs/data_{src_id}_loop"
        dir_t = REPO_ROOT / f"outputs/data_{tgt_id}_loop"

        colors_s = torch.load(dir_s / "colors.pt", map_location="cpu", weights_only=True)
        colors_t = torch.load(dir_t / "colors.pt", map_location="cpu", weights_only=True)
        world_s = torch.load(dir_s / "world_points.pt", map_location="cpu", weights_only=True)
        world_t = torch.load(dir_t / "world_points.pt", map_location="cpu", weights_only=True)
        conf_s = torch.load(dir_s / "confidence.pt", map_location="cpu", weights_only=True)
        conf_t = torch.load(dir_t / "confidence.pt", map_location="cpu", weights_only=True)

        idx_s = np.arange(0, len(colors_s), stride)
        idx_t = np.arange(0, len(colors_t), stride)

        t_ret0 = time.time()
        desc_s = compute_descriptors(colors_s[idx_s].numpy(), self.retrieval_cfg, self.device)
        desc_t = compute_descriptors(colors_t[idx_t].numpy(), self.retrieval_cfg, self.device)
        sim = desc_s @ desc_t.T
        retrieval_time_ms = (time.time() - t_ret0) * 1000

        top_flat = np.argsort(-sim, axis=None)[: top_k * 2]
        candidate_pairs = []
        seen = set()
        for flat in top_flat:
            i, j = np.unravel_index(flat, sim.shape)
            fs, ft = int(idx_s[i]), int(idx_t[j])
            if fs not in seen and len(candidate_pairs) < top_k:
                candidate_pairs.append((fs, ft, float(sim[i, j])))
                seen.add(fs)

        pair_solutions = []
        # Match keypoint pairs
        for fs, ft, score in candidate_pairs:
            t_p0 = time.time()
            img_s = colors_s[fs].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            img_t = colors_t[ft].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

            with torch.no_grad():
                feat_s = self.extractor.extract(img_s)
                feat_t = self.extractor.extract(img_t)
                match_out = self.matcher({"image0": feat_s, "image1": feat_t})
                feat_s, feat_t, match_out = [rbd(x) for x in [feat_s, feat_t, match_out]]

            matches = match_out["matches"]
            if len(matches) < 6:
                continue

            kpts_s = feat_s["keypoints"][matches[:, 0]].cpu().numpy()
            kpts_t = feat_t["keypoints"][matches[:, 1]].cpu().numpy()

            p_s, p_t = [], []
            for (xa, ya), (xb, yb) in zip(kpts_s, kpts_t):
                ia_y = min(max(int(round(ya)), 0), 279)
                ia_x = min(max(int(round(xa)), 0), 503)
                ib_y = min(max(int(round(yb)), 0), 279)
                ib_x = min(max(int(round(xb)), 0), 503)
                if conf_s[fs, ia_y, ia_x] > 0.05 and conf_t[ft, ib_y, ib_x] > 0.05:
                    pa = world_s[fs, ia_y, ia_x].numpy()
                    pb = world_t[ft, ib_y, ib_x].numpy()
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        p_s.append(pa)
                        p_t.append(pb)

            p_s, p_t = np.array(p_s), np.array(p_t)
            if len(p_s) >= 4:
                t_u0 = time.time()
                res = ransac_umeyama(p_s, p_t, estimate_scale=True, iters=3000, thresh=ransac_thresh)
                u_ms = (time.time() - t_u0) * 1000
                if res is not None:
                    s, R, t_vec, inliers, rmse = res
                    inlier_ratio = len(inliers) / len(p_s)
                    pair_solutions.append({
                        "fs": fs,
                        "ft": ft,
                        "sim": score,
                        "matches": len(matches),
                        "pairs": len(p_s),
                        "inliers": len(inliers),
                        "inlier_ratio": inlier_ratio,
                        "scale": s,
                        "R": R,
                        "t": t_vec,
                        "rmse": rmse,
                        "match_time_ms": (time.time() - t_p0) * 1000,
                        "umeyama_time_ms": u_ms,
                    })

        if not pair_solutions:
            raise RuntimeError(f"No valid Umeyama solution found between Seq {src_id} and Seq {tgt_id}!")

        pair_solutions.sort(key=lambda x: (-x["inliers"], -x["inlier_ratio"], x["rmse"]))
        best = pair_solutions[0]
        total_ms = (time.time() - t0) * 1000

        print(
            f"  [{src_id} -> {tgt_id}] Best Pair ({best['fs']:03d}, {best['ft']:03d}): "
            f"{best['matches']} 2D matches -> {best['inliers']}/{best['pairs']} inliers "
            f"({best['inlier_ratio']*100:.1f}%), scale s={best['scale']:.4f}, "
            f"RMSE={best['rmse']*1000:.1f}mm | {total_ms:.1f}ms"
        )
        return {
            "src": src_id,
            "tgt": tgt_id,
            "best_pair": best,
            "all_solutions": pair_solutions,
            "retrieval_time_ms": retrieval_time_ms,
            "total_time_ms": total_ms,
        }


def run_fusion() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("Method 2: Multi-Video Stream 3D Point Cloud Fusion (05, 06, 07, 08)")
    print("  (ALIKED + LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp VGICP)")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Merge voxel size:  {args.merge_voxel_size}m")
    print(f"Output directory:  {args.output_dir}\n")

    t_total_start = time.time()

    # Initialize matcher
    matcher = MultimodalMatcher(device)

    # 1. Match Sequence 06 -> 07 (Direct overlap between C and D)
    print("\n[Step 1] Pairwise Matching: 06 -> 07...")
    res_06_07 = matcher.match_sequences("06", "07", stride=args.keyframe_stride, top_k=args.top_k_pairs)

    # 2. Match Sequence 08 -> 07 (Direct overlap between C and D)
    print("\n[Step 2] Pairwise Matching: 08 -> 07...")
    res_08_07 = matcher.match_sequences("08", "07", stride=args.keyframe_stride, top_k=args.top_k_pairs)

    # 3. Match Sequence 05 -> 06 (Overlap at Point C landmark / bookcase)
    print("\n[Step 3] Pairwise Matching: 05 -> 06...")
    res_05_06 = matcher.match_sequences("05", "06", stride=args.keyframe_stride, top_k=args.top_k_pairs)

    # Load downsampled point clouds for small_gicp refinement
    print("\n[Step 4] Loading point clouds for registration...")
    pcd_raw = {}
    pcd_down = {}
    for sid in ["05", "06", "07", "08"]:
        path = REPO_ROOT / f"outputs/data_{sid}_loop/reconstruction.ply"
        pcd = o3d.io.read_point_cloud(str(path))
        pcd_raw[sid] = pcd
        pcd_down[sid] = pcd.voxel_down_sample(0.03)
        print(f"  Seq {sid}: {len(pcd.points):,} raw points -> {len(pcd_down[sid].points):,} downsampled (3cm)")

    # Sequence 07 is the Anchor (identity)
    s_07 = 1.0
    T_07 = np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------
    # Align 06 -> 07
    # -------------------------------------------------------------
    print("\n[Step 5] Aligning 06 -> 07 (VGICP refinement)...")
    b06 = res_06_07["best_pair"]
    s_06 = b06["scale"]
    pts_06_coarse = (np.asarray(pcd_down["06"].points, dtype=np.float64) * s_06) @ b06["R"].T + b06["t"]
    gicp_06 = small_gicp.align(
        target_points=np.asarray(pcd_down["07"].points, dtype=np.float64),
        source_points=pts_06_coarse,
        init_T_target_source=np.eye(4, dtype=np.float64),
        registration_type="VGICP",
        voxel_resolution=0.08,
        downsampling_resolution=0.03,
        max_correspondence_distance=0.08,
        max_iterations=40,
        num_threads=8,
    )
    T_06_rigid = np.eye(4, dtype=np.float64)
    T_06_rigid[:3, :3] = b06["R"]
    T_06_rigid[:3, 3] = b06["t"]
    T_06_total = gicp_06.T_target_source @ T_06_rigid

    eval_06_eval = o3d.geometry.PointCloud(pcd_down["06"])
    eval_06_pts = (np.asarray(eval_06_eval.points, dtype=np.float64) * s_06) @ T_06_total[:3, :3].T + T_06_total[:3, 3]
    eval_06_eval.points = o3d.utility.Vector3dVector(eval_06_pts)
    e06_5 = o3d.pipelines.registration.evaluate_registration(eval_06_eval, pcd_down["07"], 0.05)
    e06_8 = o3d.pipelines.registration.evaluate_registration(eval_06_eval, pcd_down["07"], 0.08)
    print(f"  06 -> 07 Fine Fitness (<5cm): {e06_5.fitness*100:.2f}%, (<8cm): {e06_8.fitness*100:.2f}%, RMSE: {e06_8.inlier_rmse*1000:.1f}mm")

    # -------------------------------------------------------------
    # Align 08 -> 07
    # -------------------------------------------------------------
    print("\n[Step 6] Aligning 08 -> 07 (VGICP refinement)...")
    b08 = res_08_07["best_pair"]
    s_08 = b08["scale"]
    pts_08_coarse = (np.asarray(pcd_down["08"].points, dtype=np.float64) * s_08) @ b08["R"].T + b08["t"]

    # Target is combined 07 + 06 for maximum constraint
    pcd_07_06_down = pcd_down["07"] + eval_06_eval
    gicp_08 = small_gicp.align(
        target_points=np.asarray(pcd_07_06_down.points, dtype=np.float64),
        source_points=pts_08_coarse,
        init_T_target_source=np.eye(4, dtype=np.float64),
        registration_type="VGICP",
        voxel_resolution=0.08,
        downsampling_resolution=0.03,
        max_correspondence_distance=0.08,
        max_iterations=40,
        num_threads=8,
    )
    T_08_rigid = np.eye(4, dtype=np.float64)
    T_08_rigid[:3, :3] = b08["R"]
    T_08_rigid[:3, 3] = b08["t"]
    T_08_total = gicp_08.T_target_source @ T_08_rigid

    eval_08_eval = o3d.geometry.PointCloud(pcd_down["08"])
    eval_08_pts = (np.asarray(eval_08_eval.points, dtype=np.float64) * s_08) @ T_08_total[:3, :3].T + T_08_total[:3, 3]
    eval_08_eval.points = o3d.utility.Vector3dVector(eval_08_pts)
    e08_5 = o3d.pipelines.registration.evaluate_registration(eval_08_eval, pcd_down["07"], 0.05)
    e08_8 = o3d.pipelines.registration.evaluate_registration(eval_08_eval, pcd_down["07"], 0.08)
    print(f"  08 -> 07 Fine Fitness (<5cm): {e08_5.fitness*100:.2f}%, (<8cm): {e08_8.fitness*100:.2f}%, RMSE: {e08_8.inlier_rmse*1000:.1f}mm")

    # -------------------------------------------------------------
    # Align 05 -> 07 (Chained through 06, refined against 07 + 06)
    # -------------------------------------------------------------
    print("\n[Step 7] Aligning 05 -> 07 (Chained 05->06->07 & VGICP refinement)...")
    b05 = res_05_06["best_pair"]
    s_05_6 = b05["scale"]
    pts_05_c6 = (np.asarray(pcd_down["05"].points, dtype=np.float64) * s_05_6) @ b05["R"].T + b05["t"]
    gicp_05_6 = small_gicp.align(
        target_points=np.asarray(pcd_down["06"].points, dtype=np.float64),
        source_points=pts_05_c6,
        init_T_target_source=np.eye(4, dtype=np.float64),
        registration_type="VGICP",
        voxel_resolution=0.08,
        downsampling_resolution=0.03,
        max_correspondence_distance=0.08,
        max_iterations=40,
        num_threads=8,
    )
    T_05_6_rigid = np.eye(4, dtype=np.float64)
    T_05_6_rigid[:3, :3] = b05["R"]
    T_05_6_rigid[:3, 3] = b05["t"]
    T_05_to_06_total = gicp_05_6.T_target_source @ T_05_6_rigid

    # Chained Sim(3) into 07:
    # p_7 = T_06_total @ [s_06 * p_6] = T_06_total[:3,:3] * (s_06 * (T_05_6 @ (s_05_6 * p_5))) + T_06_total[:3,3]
    s_05 = s_05_6 * s_06
    T_scaled_05_6 = T_05_to_06_total.copy()
    T_scaled_05_6[:3, 3] *= s_06
    T_05_initial = T_06_total @ T_scaled_05_6

    # Refine 05 against (07 + 06_aligned)
    pts_05_coarse = (np.asarray(pcd_down["05"].points, dtype=np.float64) * s_05) @ T_05_initial[:3, :3].T + T_05_initial[:3, 3]
    gicp_05_global = small_gicp.align(
        target_points=np.asarray(pcd_07_06_down.points, dtype=np.float64),
        source_points=pts_05_coarse,
        init_T_target_source=np.eye(4, dtype=np.float64),
        registration_type="VGICP",
        voxel_resolution=0.08,
        downsampling_resolution=0.03,
        max_correspondence_distance=0.08,
        max_iterations=40,
        num_threads=8,
    )
    T_05_total = gicp_05_global.T_target_source @ T_05_initial

    eval_05_eval = o3d.geometry.PointCloud(pcd_down["05"])
    eval_05_pts = (np.asarray(eval_05_eval.points, dtype=np.float64) * s_05) @ T_05_total[:3, :3].T + T_05_total[:3, 3]
    eval_05_eval.points = o3d.utility.Vector3dVector(eval_05_pts)
    e05_8 = o3d.pipelines.registration.evaluate_registration(eval_05_eval, pcd_07_06_down, 0.08)
    e05_5 = o3d.pipelines.registration.evaluate_registration(eval_05_eval, pcd_07_06_down, 0.05)
    print(f"  05 -> 07 (vs 07+06) Fine Fitness (<5cm): {e05_5.fitness*100:.2f}%, (<8cm): {e05_8.fitness*100:.2f}%, RMSE: {e05_8.inlier_rmse*1000:.1f}mm")

    # -------------------------------------------------------------
    # Step 8: Apply Transformations and Export Fused Point Clouds
    # -------------------------------------------------------------
    print("\n[Step 8] Applying transformations to raw point clouds...")
    transforms = {
        "07": (s_07, T_07),
        "06": (s_06, T_06_total),
        "08": (s_08, T_08_total),
        "05": (s_05, T_05_total),
    }

    # Distinct colors for each video stream (RGB in [0, 1])
    # 05: Crimson Red, 06: Emerald Green, 07: Dodger Blue, 08: Bright Gold
    palette = {
        "05": np.array([0.92, 0.20, 0.20], dtype=np.float64),  # Red
        "06": np.array([0.15, 0.80, 0.25], dtype=np.float64),  # Green
        "07": np.array([0.12, 0.56, 1.00], dtype=np.float64),  # Blue
        "08": np.array([1.00, 0.80, 0.08], dtype=np.float64),  # Gold/Yellow
    }

    aligned_pcds_normal = []
    aligned_pcds_colored = []

    for sid in ["05", "06", "07", "08"]:
        s, T = transforms[sid]
        raw = pcd_raw[sid]
        pts_scaled = (np.asarray(raw.points, dtype=np.float64) * s)
        pts_trans = (pts_scaled @ T[:3, :3].T) + T[:3, 3]

        # Normal true-color point cloud
        pcd_n = o3d.geometry.PointCloud()
        pcd_n.points = o3d.utility.Vector3dVector(pts_trans)
        pcd_n.colors = raw.colors
        aligned_pcds_normal.append(pcd_n)

        # Distinct color point cloud
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts_trans)
        color_arr = np.tile(palette[sid], (len(pts_trans), 1))
        pcd_c.colors = o3d.utility.Vector3dVector(color_arr)
        aligned_pcds_colored.append(pcd_c)

        print(f"  Seq {sid}: {len(pts_trans):,} points transformed (scale s={s:.4f})")

    # Combine into full merged point clouds
    print("\n[Step 9] Merging full point clouds...")
    t_m0 = time.time()
    full_normal = o3d.geometry.PointCloud()
    full_colored = o3d.geometry.PointCloud()
    for p_n, p_c in zip(aligned_pcds_normal, aligned_pcds_colored):
        full_normal += p_n
        full_colored += p_c
    print(f"  Total raw fused points: {len(full_normal.points):,} ({time.time() - t_m0:.2f}s)")

    # Voxel de-duplication
    print(f"\n[Step 10] Voxel de-duplication (voxel_size = {args.merge_voxel_size}m)...")
    t_d0 = time.time()
    dedup_normal = full_normal.voxel_down_sample(args.merge_voxel_size)
    dedup_colored = full_colored.voxel_down_sample(args.merge_voxel_size)
    print(f"  De-duplicated normal points:  {len(dedup_normal.points):,} ({time.time() - t_d0:.2f}s)")
    print(f"  De-duplicated colored points: {len(dedup_colored.points):,}")

    # File paths
    normal_full_path = args.output_dir / "data_05_08_method2_full_merged.ply"
    normal_dedup_path = args.output_dir / "data_05_08_method2_merged.ply"
    colored_full_path = args.output_dir / "data_05_08_method2_colored_full_merged.ply"
    colored_dedup_path = args.output_dir / "data_05_08_method2_colored_merged.ply"
    transform_json_path = args.output_dir / "data_05_08_method2_transform.json"

    print("\n[Step 11] Writing PLY deliverable files to disk...")
    o3d.io.write_point_cloud(str(normal_full_path), full_normal)
    print(f"  Saved: {normal_full_path} ({normal_full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    o3d.io.write_point_cloud(str(normal_dedup_path), dedup_normal)
    print(f"  Saved: {normal_dedup_path} ({normal_dedup_path.stat().st_size / 1024 / 1024:.1f} MB)")

    o3d.io.write_point_cloud(str(colored_full_path), full_colored)
    print(f"  Saved: {colored_full_path} ({colored_full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    o3d.io.write_point_cloud(str(colored_dedup_path), dedup_colored)
    print(f"  Saved: {colored_dedup_path} ({colored_dedup_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save detailed transform JSON
    result_meta = {
        "method": "Method 2: Video-Assisted Multimodal Sim(3) Registration (LightGlue + 2D-to-3D + Umeyama + small_gicp)",
        "sequences": ["05", "06", "07", "08"],
        "anchor_sequence": "07",
        "merge_voxel_size_m": args.merge_voxel_size,
        "total_raw_points": len(full_normal.points),
        "dedup_points": len(dedup_normal.points),
        "color_palette": {
            "05": {"name": "Crimson Red", "rgb": [235, 50, 50], "description": "05: A -> B -> C -> B -> A 完整循环走廊"},
            "06": {"name": "Emerald Green", "rgb": [40, 200, 60], "description": "06: C -> D 右侧路线"},
            "07": {"name": "Dodger Blue (Anchor)", "rgb": [30, 144, 255], "description": "07: B -> D 全程主干路线 (基准参考系)"},
            "08": {"name": "Bright Gold", "rgb": [255, 200, 20], "description": "08: C -> D 左侧路线"},
        },
        "pairwise_alignments": {
            "06_to_07": {
                "type": "Direct Method 2 (LightGlue + Umeyama + small_gicp)",
                "selected_pair": {"frame_06": b06["fs"], "frame_07": b06["ft"], "inliers": b06["inliers"], "total_3d_pairs": b06["pairs"]},
                "scale_s": round(float(s_06), 6),
                "transform_matrix": T_06_total.tolist(),
                "fitness_5cm": round(float(e06_5.fitness), 4),
                "fitness_8cm": round(float(e06_8.fitness), 4),
                "rmse_8cm_mm": round(float(e06_8.inlier_rmse * 1000), 2),
            },
            "08_to_07": {
                "type": "Direct Method 2 (LightGlue + Umeyama + small_gicp)",
                "selected_pair": {"frame_08": b08["fs"], "frame_07": b08["ft"], "inliers": b08["inliers"], "total_3d_pairs": b08["pairs"]},
                "scale_s": round(float(s_08), 6),
                "transform_matrix": T_08_total.tolist(),
                "fitness_5cm": round(float(e08_5.fitness), 4),
                "fitness_8cm": round(float(e08_8.fitness), 4),
                "rmse_8cm_mm": round(float(e08_8.inlier_rmse * 1000), 2),
            },
            "05_to_07": {
                "type": "Chained Method 2 (05 -> 06 -> 07 & small_gicp joint refinement)",
                "selected_pair_05_06": {"frame_05": b05["fs"], "frame_06": b05["ft"], "inliers": b05["inliers"], "total_3d_pairs": b05["pairs"]},
                "scale_s": round(float(s_05), 6),
                "transform_matrix": T_05_total.tolist(),
                "fitness_5cm": round(float(e05_5.fitness), 4),
                "fitness_8cm": round(float(e05_8.fitness), 4),
                "rmse_8cm_mm": round(float(e05_8.inlier_rmse * 1000), 2),
            },
        },
        "deliverables": {
            "normal_full_merged": str(normal_full_path),
            "normal_dedup_merged": str(normal_dedup_path),
            "colored_full_merged": str(colored_full_path),
            "colored_dedup_merged": str(colored_dedup_path),
            "transform_json": str(transform_json_path),
        },
        "total_wall_time_s": round(time.time() - t_total_start, 2),
    }

    with open(transform_json_path, "w", encoding="utf-8") as f:
        json.dump(result_meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {transform_json_path}")

    print("\n" + "=" * 80)
    print(f"Method 2 Fusion Completed Successfully in {time.time() - t_total_start:.2f}s!")
    print(f"  Total Fused Raw Points:         {len(full_normal.points):,}")
    print(f"  De-duplicated Points (1.5cm):   {len(dedup_normal.points):,}")
    print(f"  Normal Fused PLY:               {normal_dedup_path}")
    print(f"  Distinct Color Fused PLY:       {colored_dedup_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_fusion()
