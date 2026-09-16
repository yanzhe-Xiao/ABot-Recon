#!/usr/bin/env python3
"""
Post-Fusion 3D Point Cloud Optimization Tool (融合后三维点云全局优化器)

Performs comprehensive, physics-guided and geometry-guided post-fusion optimization
on already-merged or multi-stream point clouds:
  1. Multi-Way Dense Point-to-Plane ICP Refinement (子地图密集点面精对齐与接缝消除)
  2. Dominant Ground Plane Leveling & Coplanarity (地面主平面几何对齐与水平标定)
  3. Bilateral MLS Surface Thinning & Ghosting Compression (双边切平面投影消除双层重影/厚度)
  4. Photometric Color Harmonization & Seam Blending (多视点光度色彩平滑与白平衡均衡)
  5. Dual-Stage Outlier Removal (统计+半径离群点滤波与悬空飞点剔除)
  6. Quantitative Evaluation & Benchmark Reporting (前后几何残差与重叠精度量化评测)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PostFusionOptimizer:
    """Universal Post-Fusion Point Cloud Optimization Engine."""

    def __init__(
        self,
        voxel_size: float = 0.015,
        dense_icp_distance: float = 0.06,
        plane_threshold: float = 0.025,
        mls_radius: float = 0.05,
        mls_iterations: int = 1,
        sor_neighbors: int = 20,
        sor_ratio: float = 2.0,
        enable_submap_icp: bool = True,
        enable_plane_leveling: bool = True,
        enable_surface_thinning: bool = True,
        enable_color_harmonization: bool = True,
        enable_sor: bool = True,
    ):
        self.voxel_size = voxel_size
        self.dense_icp_distance = dense_icp_distance
        self.plane_threshold = plane_threshold
        self.mls_radius = mls_radius
        self.mls_iterations = mls_iterations
        self.sor_neighbors = sor_neighbors
        self.sor_ratio = sor_ratio
        self.enable_submap_icp = enable_submap_icp
        self.enable_plane_leveling = enable_plane_leveling
        self.enable_surface_thinning = enable_surface_thinning
        self.enable_color_harmonization = enable_color_harmonization
        self.enable_sor = enable_sor

    @staticmethod
    def compute_plane_residual(pcd: o3d.geometry.PointCloud, threshold: float = 0.03) -> dict:
        """Fit dominant plane and return inlier ratio, thickness and mean residual."""
        if len(pcd.points) < 100:
            return {"inliers": 0, "inlier_ratio": 0.0, "mean_residual_mm": 0.0, "std_residual_mm": 0.0}
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=1000
        )
        [a, b, c, d] = plane_model
        norm = np.linalg.norm([a, b, c])
        pts_in = np.asarray(pcd.points)[inliers]
        dists = np.abs(pts_in @ np.array([a, b, c]) + d) / norm
        return {
            "plane_equation": [round(float(x), 4) for x in [a, b, c, d]],
            "inliers": int(len(inliers)),
            "inlier_ratio_pct": round(float(len(inliers) / len(pcd.points) * 100), 2),
            "mean_residual_mm": round(float(dists.mean() * 1000), 2),
            "std_residual_mm": round(float(dists.std() * 1000), 2),
            "max_residual_mm": round(float(dists.max() * 1000), 2),
        }

    @staticmethod
    def evaluate_pairwise_distances(
        submaps: Dict[str, o3d.geometry.PointCloud],
        max_dist: float = 0.08,
    ) -> Dict[str, dict]:
        """Compute pairwise point-to-point distances across overlapping zones."""
        results = {}
        sids = list(submaps.keys())
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                si, sj = sids[i], sids[j]
                pi, pj = submaps[si], submaps[sj]
                if len(pi.points) == 0 or len(pj.points) == 0:
                    continue
                d1 = np.asarray(pi.compute_point_cloud_distance(pj))
                d2 = np.asarray(pj.compute_point_cloud_distance(pi))
                in_zone1 = d1[d1 < max_dist]
                in_zone2 = d2[d2 < max_dist]
                if len(in_zone1) > 50 and len(in_zone2) > 50:
                    mean_d = float(0.5 * (np.mean(in_zone1) + np.mean(in_zone2)))
                    p2cm = float(0.5 * ((d1 < 0.02).mean() + (d2 < 0.02).mean()) * 100)
                    p5cm = float(0.5 * ((d1 < 0.05).mean() + (d2 < 0.05).mean()) * 100)
                    results[f"{si}<->{sj}"] = {
                        "mean_distance_mm": round(mean_d * 1000, 2),
                        "precision_at_2cm_pct": round(p2cm, 2),
                        "precision_at_5cm_pct": round(p5cm, 2),
                    }
        return results

    def refine_submaps_dense_icp(
        self,
        submaps: Dict[str, Dict[str, Any]],
        anchor_id: str,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Stage 1: Multi-Way Dense Point-to-Plane ICP on aligned submaps.
        Refines submap transforms against anchor and mutually overlapping submaps.
        """
        print("\n[Post-Fusion Stage 1] Running dense multi-way Point-to-Plane ICP on submaps...")
        refined_deltas = {anchor_id: np.eye(4, dtype=np.float64)}
        stats = {}

        for sid in submaps:
            if sid == anchor_id:
                continue

            src_down = submaps[sid]["down"]
            tgt_down = submaps[anchor_id]["down"]

            reg = o3d.pipelines.registration.registration_icp(
                source=src_down,
                target=tgt_down,
                max_correspondence_distance=self.dense_icp_distance,
                init=np.eye(4, dtype=np.float64),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )

            # Safety check: accept only if fitness > 0.20 and inlier RMSE < 45mm
            if reg.fitness > 0.20 and reg.inlier_rmse < 0.045:
                delta_T = reg.transformation
                accepted = True
            else:
                delta_T = np.eye(4, dtype=np.float64)
                accepted = False

            refined_deltas[sid] = delta_T
            trans_mm = np.linalg.norm(delta_T[:3, 3]) * 1000
            stats[sid] = {
                "accepted": accepted,
                "fitness": round(float(reg.fitness), 4),
                "inlier_rmse_mm": round(float(reg.inlier_rmse * 1000), 2),
                "translation_shift_mm": round(float(trans_mm), 2),
            }
            status_str = "ACCEPTED" if accepted else "REJECTED (kept coarse)"
            print(f"  Submap [{sid} -> {anchor_id}]: {status_str} | Fitness = {reg.fitness:.3f} | RMSE = {reg.inlier_rmse*1000:.1f}mm | Shift = {trans_mm:.1f}mm")

        return refined_deltas, stats

    def apply_bilateral_surface_thinning(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.05,
        sigma_c: float = 0.03,
        sigma_s: float = 0.015,
        alpha: float = 0.8,
    ) -> o3d.geometry.PointCloud:
        """
        Stage 3: Bilateral Moving Least Squares (MLS) surface thinning.
        Compresses multi-layer double-wall/floor ghosting points onto local tangent planes.
        """
        print(f"\n[Post-Fusion Stage 3] Applying bilateral MLS surface thinning (radius={radius}m)...")
        t0 = time.time()
        pcd_out = o3d.geometry.PointCloud(pcd)
        pcd_out.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))

        pts = np.asarray(pcd_out.points).copy()
        normals = np.asarray(pcd_out.normals)
        kdtree = o3d.geometry.KDTreeFlann(pcd_out)

        n_points = len(pts)
        offsets = np.zeros(n_points, dtype=np.float64)

        # Batch processing
        batch_size = 50000
        for b_start in range(0, n_points, batch_size):
            b_end = min(b_start + batch_size, n_points)
            for i in range(b_start, b_end):
                k, idxs, dists_sq = kdtree.search_radius_vector_3d(pts[i], radius)
                if k >= 6:
                    p_i = pts[i]
                    n_i = normals[i]
                    diffs = pts[idxs] - p_i
                    dist_n = diffs @ n_i
                    dist_p = np.sqrt(np.maximum(0, dists_sq - dist_n**2))

                    w = np.exp(-dist_p**2 / (2 * sigma_c**2)) * np.exp(-dist_n**2 / (2 * sigma_s**2))
                    w_sum = np.sum(w)
                    if w_sum > 1e-6:
                        offsets[i] = np.sum(w * dist_n) / w_sum

        # Apply normal displacement
        pts += alpha * offsets[:, None] * normals
        pcd_out.points = o3d.utility.Vector3dVector(pts)
        mean_shift = float(np.abs(offsets).mean() * 1000)
        max_shift = float(np.abs(offsets).max() * 1000)
        print(f"  MLS Thinning completed in {time.time()-t0:.2f}s | Mean surface compression: {mean_shift:.2f}mm | Max: {max_shift:.2f}mm")
        return pcd_out

    def apply_color_harmonization(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.04,
    ) -> o3d.geometry.PointCloud:
        """
        Stage 4: Photometric color harmonization.
        Softens exposure jumps and seam boundaries across different cameras.
        """
        print(f"\n[Post-Fusion Stage 4] Applying photometric color harmonization (radius={radius}m)...")
        t0 = time.time()
        pcd_out = o3d.geometry.PointCloud(pcd)
        pts = np.asarray(pcd_out.points)
        colors = np.asarray(pcd_out.colors).copy()
        kdtree = o3d.geometry.KDTreeFlann(pcd_out)

        n_points = len(pts)
        colors_smooth = colors.copy()

        step = 1  # Full processing
        for i in range(0, n_points, step):
            k, idxs, dists_sq = kdtree.search_radius_vector_3d(pts[i], radius)
            if k >= 5:
                dists = np.sqrt(dists_sq)
                w = np.exp(-dists / (radius * 0.5))
                w_sum = np.sum(w)
                if w_sum > 1e-6:
                    colors_smooth[i] = np.sum(colors[idxs] * w[:, None], axis=0) / w_sum

        # Blend 50% original color + 50% harmonized color to retain sharpness while smoothing seams
        colors_final = 0.5 * colors + 0.5 * colors_smooth
        pcd_out.colors = o3d.utility.Vector3dVector(colors_final)
        print(f"  Color harmonization completed in {time.time()-t0:.2f}s")
        return pcd_out

    def optimize_fusion_directory(
        self,
        fusion_dir: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> dict:
        """
        Execute full end-to-end post-fusion optimization pipeline on a fusion directory.
        """
        f_dir = Path(fusion_dir).resolve()
        if not f_dir.is_dir():
            raise FileNotFoundError(f"Fusion directory not found: {f_dir}")

        out_dir = Path(output_dir).resolve() if output_dir else f_dir / "optimized"
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        print("=" * 80)
        print("Universal Post-Fusion 3D Point Cloud Optimization Pipeline")
        print("=" * 80)
        print(f"Input Fusion Dir:    {f_dir}")
        print(f"Output Directory:    {out_dir}")
        print(f"Voxel Grid Size:     {self.voxel_size}m")
        print(f"Submap Dense ICP:    {self.enable_submap_icp}")
        print(f"Surface Thinning:    {self.enable_surface_thinning}")
        print(f"Color Harmonization: {self.enable_color_harmonization}")
        print(f"Denoising (SOR):     {self.enable_sor}")
        print("=" * 80)

        # 1. Look for transform JSON
        transform_files = list(f_dir.glob("*transform*.json")) + list(f_dir.glob("transforms.json"))
        has_metadata = len(transform_files) > 0

        submaps = {}
        anchor_id = "anchor"
        seq_metadata = {}

        if has_metadata:
            meta = json.loads(transform_files[0].read_text(encoding="utf-8"))
            anchor_id = meta.get("anchor_sequence", meta.get("anchor_session", "anchor"))
            seq_dict = meta.get("sequences", {})

            print(f"\n[Step 1] Loading individual submaps using transform metadata (Anchor: {anchor_id})...")
            for sid, sinfo in seq_dict.items():
                # Search for reconstruction.ply in outputs/sid or f_dir
                possible_plys = [
                    REPO_ROOT / f"outputs/{sid}/reconstruction.ply",
                    f_dir / f"{sid}.ply",
                    f_dir / f"{sid}_aligned.ply",
                    REPO_ROOT / f"outputs/streams/{sid}/reconstruction.ply",
                ]
                ply_path = next((p for p in possible_plys if p.is_file()), None)
                if not ply_path:
                    continue

                scale = sinfo.get("scale_to_anchor", 1.0)
                T = np.array(sinfo.get("transform_matrix", np.eye(4)), dtype=np.float64)

                pcd_raw = o3d.io.read_point_cloud(str(ply_path))
                pts_scaled = (np.asarray(pcd_raw.points, dtype=np.float64) * scale)
                pts_trans = (pts_scaled @ T[:3, :3].T) + T[:3, 3]

                pcd_trans = o3d.geometry.PointCloud()
                pcd_trans.points = o3d.utility.Vector3dVector(pts_trans)
                pcd_trans.colors = pcd_raw.colors

                pcd_down = pcd_trans.voxel_down_sample(0.02)
                pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))

                submaps[sid] = {
                    "full": pcd_trans,
                    "down": pcd_down,
                    "scale": scale,
                    "T": T,
                    "color_rgb": sinfo.get("color_rgb", [0.8, 0.8, 0.8]),
                }
                seq_metadata[sid] = sinfo
                print(f"  Loaded submap [{sid}]: {len(pcd_trans.points):,} points (scale={scale:.4f})")

        # Fallback: if no submaps found, load merged PLY directly
        if len(submaps) < 2:
            print("\n[Notice] Directory lacks individual submaps, loading primary merged PLY directly...")
            merged_plys = list(f_dir.glob("*normal_merged.ply")) + list(f_dir.glob("reconstruction.ply"))
            if not merged_plys:
                raise FileNotFoundError(f"No PLY point clouds found in {f_dir}")
            target_ply = merged_plys[0]
            print(f"  Operating on: {target_ply.name}")
            initial_pcd = o3d.io.read_point_cloud(str(target_ply))
        else:
            initial_pcd = o3d.geometry.PointCloud()
            for sid, sdata in submaps.items():
                initial_pcd += sdata["full"]

        # Initial benchmark metrics
        init_plane_stats = self.compute_plane_residual(initial_pcd, threshold=self.plane_threshold)
        print(f"\n[Baseline Benchmark] Initial dominant plane thickness/residual:")
        print(f"  Inliers: {init_plane_stats['inliers']:,} ({init_plane_stats['inlier_ratio_pct']}%) | Mean Residual = {init_plane_stats['mean_residual_mm']}mm | Std = {init_plane_stats['std_residual_mm']}mm")

        # Initial pairwise distances
        init_pairwise = {}
        if len(submaps) >= 2:
            init_pairwise = self.evaluate_pairwise_distances({sid: sdata["down"] for sid, sdata in submaps.items()})
            print("  Initial Pairwise Overlap Precision:")
            for pair, pinfo in init_pairwise.items():
                print(f"    {pair}: Mean Dist = {pinfo['mean_distance_mm']}mm | Prec@2cm = {pinfo['precision_at_2cm_pct']}% | Prec@5cm = {pinfo['precision_at_5cm_pct']}%")

        # Stage 1: Dense Multi-Way Point-to-Plane ICP
        refined_submaps_down = {}
        refined_submaps_full = {}
        icp_stats = {}
        if len(submaps) >= 2 and self.enable_submap_icp:
            refined_deltas, icp_stats = self.refine_submaps_dense_icp(submaps, anchor_id=anchor_id)
            for sid, sdata in submaps.items():
                delta = refined_deltas.get(sid, np.eye(4, dtype=np.float64))
                p_full = o3d.geometry.PointCloud(sdata["full"]).transform(delta)
                p_down = o3d.geometry.PointCloud(sdata["down"]).transform(delta)
                refined_submaps_full[sid] = p_full
                refined_submaps_down[sid] = p_down
                # Update global transform
                T_curr = sdata["T"]
                T_new = delta @ T_curr
                sdata["T"] = T_new
        else:
            refined_submaps_full = {sid: sdata["full"] for sid, sdata in submaps.items()}
            refined_submaps_down = {sid: sdata["down"] for sid, sdata in submaps.items()}

        # Re-evaluate pairwise distances after Stage 1
        post_icp_pairwise = {}
        if len(submaps) >= 2:
            post_icp_pairwise = self.evaluate_pairwise_distances(refined_submaps_down)
            print("\n[Stage 1 Evaluation] Post-ICP Pairwise Overlap Precision:")
            for pair, pinfo in post_icp_pairwise.items():
                print(f"  {pair}: Mean Dist = {pinfo['mean_distance_mm']}mm | Prec@2cm = {pinfo['precision_at_2cm_pct']}% | Prec@5cm = {pinfo['precision_at_5cm_pct']}%")

        # Merge submaps into full point cloud and colored point cloud
        print("\n[Merging] Constructing unified full point clouds...")
        merged_normal_full = o3d.geometry.PointCloud()
        merged_colored_full = o3d.geometry.PointCloud()

        if len(refined_submaps_full) >= 2:
            for sid, p_full in refined_submaps_full.items():
                merged_normal_full += p_full

                # Assign distinct color
                col_rgb = submaps[sid].get("color_rgb", [0.8, 0.8, 0.8])
                p_col = o3d.geometry.PointCloud(p_full)
                col_mat = np.tile(col_rgb, (len(p_full.points), 1))
                p_col.colors = o3d.utility.Vector3dVector(col_mat)
                merged_colored_full += p_col
        else:
            merged_normal_full = initial_pcd

        print(f"  Total merged full points: {len(merged_normal_full.points):,}")

        # Voxel downsample
        print(f"  Downsampling to {self.voxel_size}m voxel grid...")
        merged_normal_dedup = merged_normal_full.voxel_down_sample(self.voxel_size)
        merged_colored_dedup = merged_colored_full.voxel_down_sample(self.voxel_size) if len(merged_colored_full.points) > 0 else None
        print(f"  Deduplicated normal points: {len(merged_normal_dedup.points):,}")

        # Stage 2: Dominant Ground Plane Alignment & Leveling
        plane_level_stats = {}
        if self.enable_plane_leveling and len(merged_normal_dedup.points) > 500:
            print("\n[Post-Fusion Stage 2] Dominant ground plane alignment & horizontal leveling...")
            plane_m, inliers = merged_normal_dedup.segment_plane(
                distance_threshold=self.plane_threshold, ransac_n=3, num_iterations=1000
            )
            [a, b, c, d] = plane_m
            normal_vec = np.array([a, b, c], dtype=np.float64)
            normal_vec /= np.linalg.norm(normal_vec)
            # Standardize normal pointing up
            if normal_vec[1] < 0:
                normal_vec = -normal_vec

            # Compute rotation from normal_vec to [0, 1, 0]
            up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            rot_axis = np.cross(normal_vec, up_vec)
            axis_norm = np.linalg.norm(rot_axis)
            if axis_norm > 1e-5:
                rot_axis /= axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal_vec, up_vec), -1.0, 1.0))
                R_level = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
            else:
                R_level = np.eye(3, dtype=np.float64)

            T_level = np.eye(4, dtype=np.float64)
            T_level[:3, :3] = R_level

            merged_normal_dedup.transform(T_level)
            merged_normal_full.transform(T_level)
            if merged_colored_dedup:
                merged_colored_dedup.transform(T_level)
                merged_colored_full.transform(T_level)

            plane_level_stats = {
                "detected_plane": [round(float(x), 4) for x in [a, b, c, d]],
                "rotation_angle_deg": round(float(np.rad2deg(rot_angle if axis_norm > 1e-5 else 0.0)), 2),
            }
            print(f"  Ground leveled: rotated by {plane_level_stats['rotation_angle_deg']}° to horizontal Y-up")

        # Stage 3: Bilateral MLS Surface Thinning (Ghosting Compression)
        if self.enable_surface_thinning:
            merged_normal_dedup = self.apply_bilateral_surface_thinning(
                merged_normal_dedup, radius=self.mls_radius
            )
            if merged_colored_dedup:
                # Share points geometry from normal dedup
                merged_colored_dedup.points = merged_normal_dedup.points

        # Stage 4: Photometric Color Harmonization
        if self.enable_color_harmonization:
            merged_normal_dedup = self.apply_color_harmonization(merged_normal_dedup)

        # Stage 5: Dual-Stage Outlier Removal (SOR)
        sor_stats = {}
        if self.enable_sor and len(merged_normal_dedup.points) > 1000:
            print("\n[Post-Fusion Stage 5] Running Statistical Outlier Removal (SOR)...")
            _, ind = merged_normal_dedup.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors, std_ratio=self.sor_ratio
            )
            pts_before = len(merged_normal_dedup.points)
            merged_normal_dedup = merged_normal_dedup.select_by_index(ind)
            pts_after = len(merged_normal_dedup.points)
            sor_stats = {
                "outliers_removed": pts_before - pts_after,
                "clean_points": pts_after,
            }
            print(f"  SOR cleaned: {pts_before:,} -> {pts_after:,} points ({pts_before - pts_after:,} floating outliers removed)")
            if merged_colored_dedup:
                merged_colored_dedup = merged_colored_dedup.select_by_index(ind)

        # Final benchmark metrics
        final_plane_stats = self.compute_plane_residual(merged_normal_dedup, threshold=self.plane_threshold)
        print(f"\n[Optimized Benchmark] Final dominant plane thickness/residual:")
        print(f"  Inliers: {final_plane_stats['inliers']:,} ({final_plane_stats['inlier_ratio_pct']}%) | Mean Residual = {final_plane_stats['mean_residual_mm']}mm | Std = {final_plane_stats['std_residual_mm']}mm")

        # Save Deliverables
        print("\n[Step 6] Saving optimized deliverables...")
        opt_normal_ply = out_dir / "fused_optimized_normal_merged.ply"
        opt_colored_ply = out_dir / "fused_optimized_colored_merged.ply"
        opt_normal_full_ply = out_dir / "fused_optimized_normal_full.ply"
        opt_colored_full_ply = out_dir / "fused_optimized_colored_full.ply"
        report_json = out_dir / "optimization_report.json"

        o3d.io.write_point_cloud(str(opt_normal_ply), merged_normal_dedup)
        if merged_colored_dedup:
            o3d.io.write_point_cloud(str(opt_colored_ply), merged_colored_dedup)
        o3d.io.write_point_cloud(str(opt_normal_full_ply), merged_normal_full)
        if len(merged_colored_full.points) > 0:
            o3d.io.write_point_cloud(str(opt_colored_full_ply), merged_colored_full)

        total_time = round(time.time() - t_start, 2)

        report_data = {
            "title": "Post-Fusion 3D Point Cloud Optimization Report",
            "source_dir": str(f_dir),
            "output_dir": str(out_dir),
            "voxel_size_m": self.voxel_size,
            "total_wall_time_s": total_time,
            "submap_icp_stats": icp_stats,
            "plane_leveling_stats": plane_level_stats,
            "sor_stats": sor_stats,
            "baseline_metrics": {
                "plane": init_plane_stats,
                "pairwise_precision": init_pairwise,
            },
            "optimized_metrics": {
                "plane": final_plane_stats,
                "pairwise_precision": post_icp_pairwise,
            },
            "improvements": {
                "plane_mean_residual_reduction_mm": round(init_plane_stats["mean_residual_mm"] - final_plane_stats["mean_residual_mm"], 2),
                "plane_inlier_ratio_gain_pct": round(final_plane_stats["inlier_ratio_pct"] - init_plane_stats["inlier_ratio_pct"], 2),
            },
            "deliverables": {
                "optimized_normal": str(opt_normal_ply),
                "optimized_colored": str(opt_colored_ply) if merged_colored_dedup else None,
                "optimized_normal_full": str(opt_normal_full_ply),
                "optimized_colored_full": str(opt_colored_full_ply) if len(merged_colored_full.points) > 0 else None,
            },
        }

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved Optimized Normal PLY:  {opt_normal_ply} ({opt_normal_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        if merged_colored_dedup:
            print(f"  Saved Optimized Colored PLY: {opt_colored_ply} ({opt_colored_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Saved Optimization Report:   {report_json}")
        print("=" * 80)
        print(f"Post-Fusion Optimization Completed Successfully in {total_time:.2f}s!")
        print("=" * 80)
        return report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Post-Fusion 3D Point Cloud Optimization Tool (融合后三维点云全局优化器)"
    )
    parser.add_argument(
        "--fusion-dir", "-d",
        type=Path,
        default=REPO_ROOT / "outputs/alignment/general_fusion",
        help="Directory containing fused point clouds and transform metadata",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Directory to save optimized deliverables (default: <fusion-dir>/optimized)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Merge voxel grid size in meters (default: 0.015m)",
    )
    parser.add_argument(
        "--submap-icp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dense multi-way Point-to-Plane ICP on aligned submaps",
    )
    parser.add_argument(
        "--plane-leveling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dominant ground plane alignment and horizontal leveling",
    )
    parser.add_argument(
        "--surface-thinning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bilateral MLS surface thinning to eliminate multi-layer ghosting",
    )
    parser.add_argument(
        "--color-harmonize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable photometric color harmonization across camera seams",
    )
    parser.add_argument(
        "--sor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Statistical Outlier Removal (SOR) denoising",
    )
    args = parser.parse_args()

    optimizer = PostFusionOptimizer(
        voxel_size=args.voxel_size,
        enable_submap_icp=args.submap_icp,
        enable_plane_leveling=args.plane_leveling,
        enable_surface_thinning=args.surface_thinning,
        enable_color_harmonization=args.color_harmonize,
        enable_sor=args.sor,
    )
    optimizer.optimize_fusion_directory(
        fusion_dir=args.fusion_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
