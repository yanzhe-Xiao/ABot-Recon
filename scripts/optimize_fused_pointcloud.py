#!/usr/bin/env python3
"""
Post-Fusion 3D Point Cloud Optimization Tool (融合后三维点云全局优化器)

Performs comprehensive, physics-guided and geometry-guided post-fusion optimization
on already-merged or multi-stream point clouds:
  1. Global Pose Graph Optimization across all submap pairs (全图位姿图优化替代星形ICP)
  2. Dominant Ground Plane Leveling & Coplanarity (地面主平面几何对齐与水平标定)
  3. Pre-Clean Statistical Outlier Removal (MLS前置统计去噪)
  4. Multi-Iteration Vectorized Bilateral MLS Surface Thinning (多次迭代向量化双边MLS消除双层重影)
  5. Post-Clean Radius Outlier Removal (MLS后置半径去噪)
  6. Submap-Aware Photometric Color Gain Correction (子图感知色彩增益校正)
  7. Quantitative Evaluation & Benchmark Reporting (前后几何残差与重叠精度量化评测)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Set thread limits to prevent OpenMP oversubscription or deadlocks in thread pools
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from abot_recon.sparse_loop.gpu_pgo import optimize_pose_graph_gpu_sparse

    _HAS_GPU_PGO = torch.cuda.is_available()
except ImportError:
    _HAS_GPU_PGO = False


class PostFusionOptimizer:
    """Universal Post-Fusion Point Cloud Optimization Engine."""

    def __init__(
        self,
        voxel_size: float = 0.015,
        dense_icp_distance: float = 0.08,
        plane_threshold: float = 0.025,
        mls_radius: float = 0.08,
        mls_iterations: int = 3,
        mls_k_neighbors: int = 50,
        sor_neighbors: int = 30,
        sor_ratio: float = 1.2,
        ror_min_neighbors: int = 12,
        ror_radius: float = 0.05,
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
        self.mls_k_neighbors = mls_k_neighbors
        self.sor_neighbors = sor_neighbors
        self.sor_ratio = sor_ratio
        self.ror_min_neighbors = ror_min_neighbors
        self.ror_radius = ror_radius
        self.enable_submap_icp = enable_submap_icp
        self.enable_plane_leveling = enable_plane_leveling
        self.enable_surface_thinning = enable_surface_thinning
        self.enable_color_harmonization = enable_color_harmonization
        self.enable_sor = enable_sor

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_plane_residual(pcd: o3d.geometry.PointCloud, threshold: float = 0.03) -> dict:
        """Fit dominant plane and return inlier ratio, thickness and mean residual."""
        if len(pcd.points) < 100:
            return {"inliers": 0, "inlier_ratio_pct": 0.0, "mean_residual_mm": 0.0, "std_residual_mm": 0.0}
        if len(pcd.points) > 200_000:
            pcd_fit = pcd.voxel_down_sample(0.02)
        else:
            pcd_fit = pcd
        plane_model, _ = pcd_fit.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=500
        )
        [a, b, c, d] = plane_model
        norm = np.linalg.norm([a, b, c])
        pts_all = np.asarray(pcd.points)
        dists_all = np.abs(pts_all @ np.array([a, b, c]) + d) / norm
        inliers_mask = dists_all <= threshold
        inliers_count = int(inliers_mask.sum())
        dists = dists_all[inliers_mask]
        return {
            "plane_equation": [round(float(x), 4) for x in [a, b, c, d]],
            "inliers": inliers_count,
            "inlier_ratio_pct": round(float(inliers_count / len(pcd.points) * 100), 2),
            "mean_residual_mm": round(float(dists.mean() * 1000), 2) if inliers_count > 0 else 0.0,
            "std_residual_mm": round(float(dists.std() * 1000), 2) if inliers_count > 0 else 0.0,
            "max_residual_mm": round(float(dists.max() * 1000), 2) if inliers_count > 0 else 0.0,
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

    # ------------------------------------------------------------------
    # Stage 1: Global Pose Graph Optimization (replaces star-topology ICP)
    # ------------------------------------------------------------------

    def _build_pairwise_icp_edges(
        self,
        submaps: Dict[str, Dict[str, Any]],
        sids: List[str],
    ) -> Tuple[List[int], List[int], List[np.ndarray], List[float], dict]:
        """Run all-pairs Point-to-Plane ICP and return PGO edge data."""
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_measurements: List[np.ndarray] = []
        edge_weights: List[float] = []
        edge_stats: dict = {}

        sid_to_idx = {s: i for i, s in enumerate(sids)}

        for i, si in enumerate(sids):
            for j, sj in enumerate(sids):
                if i >= j:
                    continue
                src_down = submaps[si]["down"]
                tgt_down = submaps[sj]["down"]

                # Multi-scale coarse-to-fine Point-to-Plane ICP:
                T_curr_ij = np.eye(4, dtype=np.float64)
                for dist, vox in [(0.20, 0.04), (0.10, 0.025), (self.dense_icp_distance, 0.015)]:
                    s_vox = src_down.voxel_down_sample(vox)
                    t_vox = tgt_down.voxel_down_sample(vox)
                    s_vox.estimate_normals()
                    t_vox.estimate_normals()
                    reg = o3d.pipelines.registration.registration_icp(
                        source=s_vox, target=t_vox, max_correspondence_distance=dist,
                        init=T_curr_ij,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    )
                    if reg.fitness > 0.15:
                        T_curr_ij = reg.transformation

                reg_ij = o3d.pipelines.registration.registration_icp(
                    source=src_down, target=tgt_down,
                    max_correspondence_distance=self.dense_icp_distance,
                    init=T_curr_ij,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                )

                # Reverse ICP: source=sj -> target=si with multi-scale
                T_curr_ji = np.eye(4, dtype=np.float64)
                for dist, vox in [(0.20, 0.04), (0.10, 0.025), (self.dense_icp_distance, 0.015)]:
                    s_vox = tgt_down.voxel_down_sample(vox)
                    t_vox = src_down.voxel_down_sample(vox)
                    s_vox.estimate_normals()
                    t_vox.estimate_normals()
                    reg = o3d.pipelines.registration.registration_icp(
                        source=s_vox, target=t_vox, max_correspondence_distance=dist,
                        init=T_curr_ji,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    )
                    if reg.fitness > 0.15:
                        T_curr_ji = reg.transformation

                reg_ji = o3d.pipelines.registration.registration_icp(
                    source=tgt_down, target=src_down,
                    max_correspondence_distance=self.dense_icp_distance,
                    init=T_curr_ji,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                )
                pair_label = f"{si}<->{sj}"

                # Forward edge (i->j) if valid
                if reg_ij.fitness > 0.10:
                    weight_ij = reg_ij.fitness / (reg_ij.inlier_rmse + 1e-3)
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_measurements.append(reg_ij.transformation.copy())
                    edge_weights.append(weight_ij)

                # Reverse edge (j->i) if valid
                if reg_ji.fitness > 0.10:
                    weight_ji = reg_ji.fitness / (reg_ji.inlier_rmse + 1e-3)
                    edge_src.append(j)
                    edge_dst.append(i)
                    edge_measurements.append(reg_ji.transformation.copy())
                    edge_weights.append(weight_ji)

                edge_stats[pair_label] = {
                    "fitness_ij": round(float(reg_ij.fitness), 4),
                    "rmse_ij_mm": round(float(reg_ij.inlier_rmse * 1000), 2),
                    "fitness_ji": round(float(reg_ji.fitness), 4),
                    "rmse_ji_mm": round(float(reg_ji.inlier_rmse * 1000), 2),
                }
                print(
                    f"    {pair_label}: "
                    f"i->j fitness={reg_ij.fitness:.3f} RMSE={reg_ij.inlier_rmse*1000:.1f}mm | "
                    f"j->i fitness={reg_ji.fitness:.3f} RMSE={reg_ji.inlier_rmse*1000:.1f}mm"
                )

        return edge_src, edge_dst, edge_measurements, edge_weights, edge_stats

    def refine_submaps_global_pgo(
        self,
        submaps: Dict[str, Dict[str, Any]],
        anchor_id: str,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Stage 1: Global Pose Graph Optimization across all submap pairs.

        Builds a full pairwise ICP edge graph and optimizes all submap poses
        simultaneously. Falls back to Open3D PGO if GPU PGO is unavailable.
        """
        print("\n[Post-Fusion Stage 1] Building full pairwise ICP graph + Global PGO...")

        # Ordered list: anchor first (index 0 is fixed in PGO)
        sids = [anchor_id] + [s for s in submaps if s != anchor_id]
        sid_to_idx = {s: i for i, s in enumerate(sids)}
        n_nodes = len(sids)

        # Build all-pairs edges
        print("  Computing all-pairs Point-to-Plane ICP...")
        edge_src, edge_dst, edge_meas, edge_weights, edge_stats = (
            self._build_pairwise_icp_edges(submaps, sids)
        )

        if len(edge_meas) == 0:
            print("  [Warning] No valid ICP edges found. Skipping PGO.")
            return {sid: np.eye(4, dtype=np.float64) for sid in sids}, {"skipped": True}

        print(f"  Built {len(edge_meas)} edges across {n_nodes} submaps.")

        # Initial poses: identity (submaps already roughly aligned)
        init_poses = np.stack([np.eye(4, dtype=np.float64)] * n_nodes)
        optimized_poses = init_poses.copy()
        pgo_backend = "none"

        # --- Attempt GPU PGO ---
        if _HAS_GPU_PGO:
            try:
                print("  Running GPU Sparse PGO (SE3)...")
                optimized_poses = optimize_pose_graph_gpu_sparse(
                    keyframe_c2w_init=init_poses,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_measurements=edge_meas,
                    edge_weights=edge_weights,
                    model="se3",
                    update_mode="all",
                    max_iterations=30,
                    lambda_init=1e-4,
                    device="cuda",
                    verbose=True,
                )
                pgo_backend = "gpu_sparse_pcg"
                print("  GPU PGO completed successfully.")
            except Exception as e:
                print(f"  [Warning] GPU PGO failed ({e}), falling back to Open3D PGO...")
                optimized_poses = init_poses.copy()

        # --- Fallback: Open3D Pose Graph Optimization ---
        if pgo_backend == "none":
            print("  Running Open3D Global Pose Graph Optimization...")
            pose_graph = o3d.pipelines.registration.PoseGraph()
            for i in range(n_nodes):
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(init_poses[i])
                )

            for k in range(len(edge_meas)):
                si_idx, sj_idx = edge_src[k], edge_dst[k]
                si_id, sj_id = sids[si_idx], sids[sj_idx]
                src_down = submaps[si_id]["down"]
                tgt_down = submaps[sj_id]["down"]
                info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    src_down, tgt_down, self.dense_icp_distance, edge_meas[k]
                )
                # uncertain=True for all edges (they are all pairwise ICP, not odometry)
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        si_idx, sj_idx, edge_meas[k], info, uncertain=True
                    )
                )

            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.dense_icp_distance,
                edge_prune_threshold=0.25,
                reference_node=0,  # anchor is node 0
            )
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option,
            )
            for i in range(n_nodes):
                optimized_poses[i] = np.array(pose_graph.nodes[i].pose, dtype=np.float64)
            pgo_backend = "open3d_lm"
            print("  Open3D PGO completed successfully.")

        # Extract deltas and report
        refined_deltas: Dict[str, np.ndarray] = {}
        stats: dict = {"backend": pgo_backend, "edges": len(edge_meas), "edge_details": edge_stats}

        for sid in sids:
            idx = sid_to_idx[sid]
            delta = optimized_poses[idx]
            refined_deltas[sid] = delta
            shift_mm = float(np.linalg.norm(delta[:3, 3]) * 1000)
            # Rotation angle
            R_delta = delta[:3, :3]
            cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = float(np.rad2deg(np.arccos(cos_angle)))
            status = "FIXED" if sid == anchor_id else "OPTIMIZED"
            print(f"  [{sid}]: {status} | Shift={shift_mm:.2f}mm | Rotation={angle_deg:.3f}°")
            stats[sid] = {
                "shift_mm": round(shift_mm, 2),
                "rotation_deg": round(angle_deg, 3),
            }

        return refined_deltas, stats

    # ------------------------------------------------------------------
    # Stage 3: Multi-Iteration Vectorized Bilateral MLS Surface Thinning
    # ------------------------------------------------------------------

    def apply_bilateral_surface_thinning(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.08,
        sigma_c: float = 0.04,
        sigma_s: float = 0.03,
        alpha: float = 0.8,
    ) -> o3d.geometry.PointCloud:
        """
        Stage 3: Multi-iteration vectorized bilateral MLS surface thinning.

        Uses scipy cKDTree + numpy broadcasting to eliminate Python-level loops.
        Multiple iterations with normal re-estimation progressively compress
        double-wall/floor ghosting onto a single surface.
        """
        n_raw_points = len(pcd.points)
        iterations = self.mls_iterations
        k_nn = min(self.mls_k_neighbors, max(10, n_raw_points - 1))

        if n_raw_points > 2_500_000:
            print(f"  [MLS Proxy] Subsampling {n_raw_points:,} points to ~1.4M for fast bilateral thinning...")
            pcd_out = pcd.voxel_down_sample(0.008)
            n_raw_points = len(pcd_out.points)
            iterations = 1
            k_nn = min(k_nn, 20)
            chunk_size = 120000
        else:
            pcd_out = o3d.geometry.PointCloud(pcd)
            if n_raw_points > 1_000_000:
                iterations = min(iterations, 2)
                k_nn = min(k_nn, 30)
                chunk_size = 100000
            else:
                chunk_size = 80000

        print(
            f"\n[Post-Fusion Stage 3] Bilateral MLS surface thinning "
            f"(points={n_raw_points:,}, radius={radius}m, {iterations} iters, k={k_nn}, chunk={chunk_size})..."
        )
        t0 = time.time()

        # Estimate normals once upfront if missing
        if not pcd_out.has_normals() or len(pcd_out.normals) != n_raw_points:
            pcd_out.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1.6, max_nn=30)
            )

        for it in range(iterations):
            t_iter = time.time()

            # Re-estimate normals only for moderate clouds and subsequent iterations
            if it > 0 and n_raw_points <= 1_000_000:
                pcd_out.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1.6, max_nn=30)
                )

            pts = np.asarray(pcd_out.points).copy()
            normals = np.asarray(pcd_out.normals).copy()
            n_points = len(pts)

            offsets = np.zeros(n_points, dtype=np.float64)
            tree = cKDTree(pts)
            for b_start in range(0, n_points, chunk_size):
                b_end = min(b_start + chunk_size, n_points)
                chunk_pts = pts[b_start:b_end]  # (C, 3)
                chunk_normals = normals[b_start:b_end]  # (C, 3)
                c_len = b_end - b_start

                # Vectorized KNN query
                dists_raw, idxs_raw = tree.query(
                    chunk_pts, k=k_nn, distance_upper_bound=radius, workers=-1
                )  # (C, k_nn)

                # Mask invalid entries (inf distance from cKDTree sentinel)
                valid = np.isfinite(dists_raw) & (dists_raw > 1e-12)  # exclude self
                # Clamp indices for safe indexing (sentinels = n_points)
                safe_idxs = np.clip(idxs_raw, 0, n_points - 1)  # (C, k_nn)

                # Neighbor points: (C, k_nn, 3)
                nbr_pts = pts[safe_idxs]
                # Diffs: (C, k_nn, 3)
                diffs = nbr_pts - chunk_pts[:, None, :]

                # Normal-direction displacement: (C, k_nn)
                dist_n = np.einsum("ijk,ik->ij", diffs, chunk_normals)
                # Tangential distance squared
                dist_sq = dists_raw ** 2
                dist_n_sq = dist_n ** 2
                dist_p_sq = np.maximum(0.0, dist_sq - dist_n_sq)

                # Bilateral weights: (C, k_nn)
                w = np.exp(-dist_p_sq / (2.0 * sigma_c ** 2)) * np.exp(
                    -dist_n_sq / (2.0 * sigma_s ** 2)
                )
                w *= valid  # zero out invalid neighbors

                # Enough-neighbors mask
                enough = valid.sum(axis=1) >= 6  # (C,)

                w_sum = w.sum(axis=1)  # (C,)
                weighted_dn = (w * dist_n).sum(axis=1)  # (C,)
                chunk_offsets = np.where(
                    enough & (w_sum > 1e-6), weighted_dn / w_sum, 0.0
                )
                offsets[b_start:b_end] = chunk_offsets

            # Apply normal displacement
            pts += alpha * offsets[:, None] * normals
            pcd_out.points = o3d.utility.Vector3dVector(pts)

            mean_shift = float(np.abs(offsets).mean() * 1000)
            max_shift = float(np.abs(offsets).max() * 1000)
            print(
                f"  Iteration {it + 1}/{iterations}: "
                f"mean shift={mean_shift:.2f}mm, max={max_shift:.2f}mm "
                f"({time.time() - t_iter:.2f}s)"
            )

        print(f"  MLS thinning total time: {time.time() - t0:.2f}s")
        return pcd_out

    # ------------------------------------------------------------------
    # Stage 4: Submap-Aware Photometric Color Gain Correction
    # ------------------------------------------------------------------

    def apply_submap_color_gain_correction(
        self,
        submaps_full: Dict[str, o3d.geometry.PointCloud],
        anchor_id: str,
        overlap_radius: float = 0.03,
    ) -> dict:
        """
        Stage 4: Submap-aware photometric color correction.

        Computes per-channel additive offset in overlap zones between each
        submap and the anchor.  Additive correction avoids amplifying
        intra-submap color variance (unlike multiplicative gain).
        """
        print(
            f"\n[Post-Fusion Stage 4] Submap-aware color correction "
            f"(overlap_radius={overlap_radius}m)..."
        )
        t0 = time.time()
        stats: dict = {}

        anchor_pcd = submaps_full[anchor_id]
        anchor_pts = np.asarray(anchor_pcd.points, dtype=np.float64)
        anchor_colors = np.asarray(anchor_pcd.colors, dtype=np.float64)

        if len(anchor_pts) == 0 or len(anchor_colors) == 0:
            print("  [Warning] Anchor has no color data. Skipping.")
            return stats

        anchor_down = anchor_pcd.voxel_down_sample(0.02)
        anchor_down_pts = np.asarray(anchor_down.points, dtype=np.float64)
        anchor_down_colors = np.asarray(anchor_down.colors, dtype=np.float64)
        anchor_tree = cKDTree(anchor_down_pts)

        for sid, pcd in submaps_full.items():
            if sid == anchor_id:
                continue

            pts = np.asarray(pcd.points, dtype=np.float64)
            colors = np.asarray(pcd.colors, dtype=np.float64)
            if len(pts) == 0 or len(colors) == 0:
                continue

            src_down = pcd.voxel_down_sample(0.02)
            src_down_pts = np.asarray(src_down.points, dtype=np.float64)
            src_down_colors = np.asarray(src_down.colors, dtype=np.float64)

            dists, idxs = anchor_tree.query(src_down_pts, k=1)
            overlap_mask = dists < overlap_radius
            n_overlap = int(overlap_mask.sum())

            if n_overlap < 100:
                print(f"  [{sid}]: Too few overlap points ({n_overlap}). Skipping.")
                continue

            src_overlap_colors = src_down_colors[overlap_mask]
            tgt_overlap_colors = anchor_down_colors[idxs[overlap_mask]]

            # Per-channel ADDITIVE offset (preserves intra-submap variance)
            src_median = np.median(src_overlap_colors, axis=0)
            tgt_median = np.median(tgt_overlap_colors, axis=0)
            offset = tgt_median - src_median
            offset = np.clip(offset, -0.3, 0.3)

            corrected = colors + offset[None, :]
            corrected = np.clip(corrected, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(corrected)

            stats[sid] = {
                "overlap_points": n_overlap,
                "offset_rgb": [round(float(o), 4) for o in offset],
            }
            print(
                f"  [{sid}]: {n_overlap:,} overlap pts | "
                f"offset R={offset[0]:+.3f} G={offset[1]:+.3f} B={offset[2]:+.3f}"
            )

        print(f"  Color correction completed in {time.time() - t0:.2f}s")
        return stats
    def apply_seam_color_blending(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.03,
        blend_ratio: float = 0.3,
    ) -> o3d.geometry.PointCloud:
        """
        Light color smoothing in seam zones only.

        Identifies points with high local color variance (seam boundary
        indicators) and applies a weighted average blending only to those
        points, preserving sharpness in uniform regions.
        """
        print(f"\n[Seam Blend] Targeted seam color smoothing (radius={radius}m, blend={blend_ratio})...")
        t0 = time.time()
        pcd_out = o3d.geometry.PointCloud(pcd)
        pts = np.asarray(pcd_out.points, dtype=np.float64)
        colors = np.asarray(pcd_out.colors, dtype=np.float64).copy()
        n_points = len(pts)

        if n_points == 0 or len(colors) == 0:
            return pcd_out

        tree = cKDTree(pts)
        k_nn = 20
        colors_blended = colors.copy()

        chunk_size = 80000
        n_seam = 0
        for b_start in range(0, n_points, chunk_size):
            b_end = min(b_start + chunk_size, n_points)
            chunk_pts = pts[b_start:b_end]
            dists_raw, idxs_raw = tree.query(
                chunk_pts, k=k_nn, distance_upper_bound=radius, workers=-1
            )
            valid = np.isfinite(dists_raw)
            safe_idxs = np.clip(idxs_raw, 0, n_points - 1)

            # Neighbor colors: (C, k, 3)
            nbr_colors = colors[safe_idxs]

            # Local color std per point: (C, 3) -> mean across channels -> (C,)
            nbr_colors_masked = nbr_colors.copy()
            nbr_colors_masked[~valid] = np.nan
            local_std = np.nanstd(nbr_colors_masked, axis=1)  # (C, 3)
            mean_std = np.nanmean(local_std, axis=1)  # (C,)

            # Seam threshold: points with high local color variance
            seam_mask = mean_std > 0.05  # ~5% color std = likely a seam

            # Distance-weighted average for seam points
            if seam_mask.any():
                dists_safe = np.where(valid, dists_raw, np.inf)
                w = np.exp(-dists_safe / (radius * 0.5))
                w *= valid
                w_sum = w.sum(axis=1, keepdims=True)
                w_sum = np.maximum(w_sum, 1e-6)
                smoothed = (nbr_colors * w[:, :, None]).sum(axis=1) / w_sum

                # Blend only seam points
                blended = (1.0 - blend_ratio) * colors[b_start:b_end] + blend_ratio * smoothed
                colors_blended[b_start:b_end] = np.where(
                    seam_mask[:, None], blended, colors[b_start:b_end]
                )
                n_seam += int(seam_mask.sum())

        pcd_out.colors = o3d.utility.Vector3dVector(np.clip(colors_blended, 0.0, 1.0))
        print(
            f"  Seam blend: {n_seam:,}/{n_points:,} seam points blended "
            f"({time.time() - t0:.2f}s)"
        )
        return pcd_out

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def optimize_fusion_directory(
        self,
        fusion_dir: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> dict:
        """
        Execute full end-to-end post-fusion optimization pipeline.

        Pipeline order (revised):
          1. Global PGO across all submap pairs
          2. Submap-aware color gain correction (before merge)
          3. Merge + voxel downsample
          4. Pre-clean SOR (before MLS)
          5. Dominant ground plane leveling
          6. Multi-iteration vectorized MLS thinning
          7. Post-clean ROR (after MLS)
          8. Save + benchmark
        """
        f_dir = Path(fusion_dir).resolve()
        if not f_dir.is_dir():
            raise FileNotFoundError(f"Fusion directory not found: {f_dir}")

        out_dir = Path(output_dir).resolve() if output_dir else f_dir / "optimized"
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        print("=" * 80)
        print("Universal Post-Fusion 3D Point Cloud Optimization Pipeline (v2)")
        print("=" * 80)
        print(f"Input Fusion Dir:    {f_dir}")
        print(f"Output Directory:    {out_dir}")
        print(f"Voxel Grid Size:     {self.voxel_size}m")
        print(f"Submap PGO:          {self.enable_submap_icp}")
        print(f"Surface Thinning:    {self.enable_surface_thinning} (iters={self.mls_iterations})")
        print(f"Color Correction:    {self.enable_color_harmonization}")
        print(f"Denoising SOR+ROR:   {self.enable_sor}")
        print("=" * 80)

        # ============================================================
        # Load submaps from transform metadata
        # ============================================================
        transform_files = list(f_dir.glob("*transform*.json")) + list(f_dir.glob("transforms.json"))
        has_metadata = len(transform_files) > 0

        submaps: Dict[str, Dict[str, Any]] = {}
        anchor_id = "anchor"
        seq_metadata: dict = {}

        if has_metadata:
            meta = json.loads(transform_files[0].read_text(encoding="utf-8"))
            anchor_id = meta.get("anchor_sequence", meta.get("anchor_session", "anchor"))
            seq_dict = meta.get("sequences", {})

            print(f"\n[Load] Loading individual submaps (Anchor: {anchor_id})...")
            for sid, sinfo in seq_dict.items():
                possible_plys = [
                    Path(sinfo["ply_path"]) if "ply_path" in sinfo and sinfo["ply_path"] else None,
                    (Path(sinfo["dir_path"]) / "reconstruction.ply") if "dir_path" in sinfo and sinfo["dir_path"] else None,
                    REPO_ROOT / f"outputs/{sid}/reconstruction.ply",
                    REPO_ROOT / f"outputs/streams/{sid}/reconstruction.ply",
                    REPO_ROOT / f"outputs/scenes/{sid}/reconstruction.ply",
                    REPO_ROOT / f"outputs/data_{sid}_loop/reconstruction.ply",
                    REPO_ROOT / f"outputs/{sid}_loop/reconstruction.ply",
                    REPO_ROOT / f"outputs/data_{sid}/reconstruction.ply",
                    f_dir / f"{sid}.ply",
                    f_dir / f"{sid}_aligned.ply",
                ]
                # Dynamic search in streams directory
                streams_dir = REPO_ROOT / "outputs/streams"
                if streams_dir.is_dir():
                    possible_plys.extend([
                        p / "reconstruction.ply" for p in streams_dir.glob(f"*{sid}*")
                        if p.is_dir() and (p / "reconstruction.ply").is_file()
                    ])
                ply_path = next((p for p in possible_plys if p is not None and Path(p).is_file()), None)
                if not ply_path:
                    continue

                scale = sinfo.get("scale_to_anchor", 1.0)
                T = np.array(sinfo.get("transform_matrix", np.eye(4)), dtype=np.float64)

                pcd_raw = o3d.io.read_point_cloud(str(ply_path))
                pts_scaled = np.asarray(pcd_raw.points, dtype=np.float64) * scale
                pts_trans = (pts_scaled @ T[:3, :3].T) + T[:3, 3]

                pcd_trans = o3d.geometry.PointCloud()
                pcd_trans.points = o3d.utility.Vector3dVector(pts_trans)
                pcd_trans.colors = pcd_raw.colors

                pcd_down = pcd_trans.voxel_down_sample(0.02)
                pcd_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
                )

                submaps[sid] = {
                    "full": pcd_trans,
                    "down": pcd_down,
                    "scale": scale,
                    "T": T,
                    "color_rgb": sinfo.get("color_rgb", [0.8, 0.8, 0.8]),
                }
                seq_metadata[sid] = sinfo
                print(f"  Loaded [{sid}]: {len(pcd_trans.points):,} pts (scale={scale:.4f})")

        # Fallback: load merged PLY directly
        if len(submaps) < 2:
            print("\n[Notice] No individual submaps found; loading merged PLY directly...")
            merged_plys = list(f_dir.glob("*normal_merged.ply")) + list(f_dir.glob("reconstruction.ply"))
            if not merged_plys:
                raise FileNotFoundError(f"No PLY point clouds found in {f_dir}")
            target_ply = merged_plys[0]
            print(f"  Operating on: {target_ply.name}")
            initial_pcd = o3d.io.read_point_cloud(str(target_ply))
        else:
            initial_pcd = o3d.geometry.PointCloud()
            for sdata in submaps.values():
                initial_pcd += sdata["full"]

        # ============================================================
        # Baseline benchmark
        # ============================================================
        init_plane_stats = self.compute_plane_residual(initial_pcd, threshold=self.plane_threshold)
        print(f"\n[Baseline] Dominant plane residual:")
        print(
            f"  Inliers: {init_plane_stats['inliers']:,} ({init_plane_stats['inlier_ratio_pct']}%) "
            f"| Mean={init_plane_stats['mean_residual_mm']}mm | Std={init_plane_stats['std_residual_mm']}mm"
        )

        init_pairwise: dict = {}
        if len(submaps) >= 2:
            init_pairwise = self.evaluate_pairwise_distances(
                {sid: sdata["down"] for sid, sdata in submaps.items()}
            )
            print("  Initial pairwise precision:")
            for pair, pinfo in init_pairwise.items():
                print(
                    f"    {pair}: Mean={pinfo['mean_distance_mm']}mm "
                    f"| @2cm={pinfo['precision_at_2cm_pct']}% | @5cm={pinfo['precision_at_5cm_pct']}%"
                )

        # ============================================================
        # Stage 1: Global PGO
        # ============================================================
        pgo_stats: dict = {}
        post_pgo_pairwise: dict = {}
        if len(submaps) >= 2 and self.enable_submap_icp:
            refined_deltas, pgo_stats = self.refine_submaps_global_pgo(submaps, anchor_id)

            for sid, sdata in submaps.items():
                delta = refined_deltas.get(sid, np.eye(4, dtype=np.float64))
                sdata["full"] = o3d.geometry.PointCloud(sdata["full"]).transform(delta)
                sdata["down"] = o3d.geometry.PointCloud(sdata["down"]).transform(delta)
                sdata["T"] = delta @ sdata["T"]

            # Post-PGO evaluation
            post_pgo_pairwise = self.evaluate_pairwise_distances(
                {sid: sdata["down"] for sid, sdata in submaps.items()}
            )
            print("\n[Stage 1 Eval] Post-PGO pairwise precision:")
            for pair, pinfo in post_pgo_pairwise.items():
                print(
                    f"  {pair}: Mean={pinfo['mean_distance_mm']}mm "
                    f"| @2cm={pinfo['precision_at_2cm_pct']}% | @5cm={pinfo['precision_at_5cm_pct']}%"
                )

        # ============================================================
        # Stage 4 (early): Submap-aware color gain correction
        # ============================================================
        color_stats: dict = {}
        if len(submaps) >= 2 and self.enable_color_harmonization:
            submaps_full_ref = {sid: sdata["full"] for sid, sdata in submaps.items()}
            color_stats = self.apply_submap_color_gain_correction(
                submaps_full_ref, anchor_id
            )

        # ============================================================
        # Merge submaps
        # ============================================================
        print("\n[Merge] Constructing unified point clouds...")
        merged_normal_full = o3d.geometry.PointCloud()
        merged_colored_full = o3d.geometry.PointCloud()

        if len(submaps) >= 2:
            for sid, sdata in submaps.items():
                p_full = sdata["full"]
                merged_normal_full += p_full

                # Distinct-color version for visualization
                p_col = o3d.geometry.PointCloud(p_full)
                col_rgb = sdata.get("color_rgb", [0.8, 0.8, 0.8])
                col_mat = np.tile(col_rgb, (len(p_full.points), 1))
                p_col.colors = o3d.utility.Vector3dVector(col_mat)
                merged_colored_full += p_col
        else:
            merged_normal_full = initial_pcd

        print(f"  Total merged points: {len(merged_normal_full.points):,}")

        # Voxel downsample
        print(f"  Downsampling to {self.voxel_size}m voxel grid...")
        merged_normal_dedup = merged_normal_full.voxel_down_sample(self.voxel_size)
        merged_colored_dedup = (
            merged_colored_full.voxel_down_sample(self.voxel_size)
            if len(merged_colored_full.points) > 0
            else None
        )
        print(f"  Deduplicated points: {len(merged_normal_dedup.points):,}")

        # ============================================================
        # Pre-clean SOR (before MLS to remove outliers that confuse MLS)
        # ============================================================
        pre_sor_stats: dict = {}
        if self.enable_sor and len(merged_normal_dedup.points) > 1000:
            print("\n[Pre-Clean] Statistical Outlier Removal (before MLS)...")
            _, ind = merged_normal_dedup.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors, std_ratio=self.sor_ratio
            )
            pts_before = len(merged_normal_dedup.points)
            merged_normal_dedup = merged_normal_dedup.select_by_index(ind)
            pts_after = len(merged_normal_dedup.points)
            pre_sor_stats = {
                "outliers_removed": pts_before - pts_after,
                "clean_points": pts_after,
            }
            print(f"  SOR: {pts_before:,} -> {pts_after:,} ({pts_before - pts_after:,} removed)")
            if merged_colored_dedup:
                merged_colored_dedup = merged_colored_dedup.select_by_index(ind)

        # ============================================================
        # Stage 2: Dominant Ground Plane Leveling
        # ============================================================
        plane_level_stats: dict = {}
        if self.enable_plane_leveling and len(merged_normal_dedup.points) > 500:
            print("\n[Stage 2] Dominant ground plane alignment & leveling...")
            if len(merged_normal_dedup.points) > 200_000:
                pcd_plane = merged_normal_dedup.voxel_down_sample(0.02)
            else:
                pcd_plane = merged_normal_dedup
            plane_m, _ = pcd_plane.segment_plane(
                distance_threshold=self.plane_threshold, ransac_n=3, num_iterations=500
            )
            [a, b, c, d] = plane_m
            normal_vec = np.array([a, b, c], dtype=np.float64)
            normal_vec /= np.linalg.norm(normal_vec)

            # Choose up-axis: use the axis with largest absolute component
            abs_components = np.abs(normal_vec)
            up_axis_idx = int(np.argmax(abs_components))
            up_vec = np.zeros(3, dtype=np.float64)
            up_vec[up_axis_idx] = 1.0

            # Ensure normal points along positive up direction
            if normal_vec[up_axis_idx] < 0:
                normal_vec = -normal_vec

            rot_axis = np.cross(normal_vec, up_vec)
            axis_norm = np.linalg.norm(rot_axis)
            if axis_norm > 1e-5:
                rot_axis /= axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal_vec, up_vec), -1.0, 1.0))
                R_level = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
            else:
                R_level = np.eye(3, dtype=np.float64)
                rot_angle = 0.0

            T_level = np.eye(4, dtype=np.float64)
            T_level[:3, :3] = R_level

            merged_normal_dedup.transform(T_level)
            merged_normal_full.transform(T_level)
            if merged_colored_dedup:
                merged_colored_dedup.transform(T_level)
                merged_colored_full.transform(T_level)

            plane_level_stats = {
                "detected_plane": [round(float(x), 4) for x in [a, b, c, d]],
                "up_axis": ["X", "Y", "Z"][up_axis_idx],
                "rotation_angle_deg": round(float(np.rad2deg(rot_angle)), 2),
            }
            print(
                f"  Ground leveled: rotated {plane_level_stats['rotation_angle_deg']}° "
                f"to {plane_level_stats['up_axis']}-up"
            )

        # ============================================================
        # Stage 2.5: Ground Sub-Surface Mirror Reflection Cutoff
        # ============================================================
        if self.enable_plane_leveling and len(merged_normal_dedup.points) > 500:
            print("\n[Stage 2.5] Ground reflection & sub-surface cutoff...")
            try:
                p_down_leveled = merged_normal_dedup.voxel_down_sample(0.02)
                plane_eq, inliers_g = p_down_leveled.segment_plane(
                    distance_threshold=self.plane_threshold, ransac_n=3, num_iterations=300
                )
                if abs(plane_eq[up_axis_idx]) > 0.6:
                    y_ground = -plane_eq[3] / plane_eq[up_axis_idx]
                    p_pts = np.asarray(merged_normal_dedup.points)
                    valid_above = p_pts[:, up_axis_idx] >= (y_ground - 0.02)
                    n_refl = int((~valid_above).sum())
                    if 0 < n_refl < len(p_pts) * 0.3:
                        idx_keep = np.where(valid_above)[0]
                        merged_normal_dedup = merged_normal_dedup.select_by_index(idx_keep)
                        if merged_colored_dedup:
                            merged_colored_dedup = merged_colored_dedup.select_by_index(idx_keep)
                        print(f"  Removed {n_refl:,} mirror reflection points below floor level (Y < {y_ground - 0.02:.3f}m)")
            except Exception as pe:
                print(f"  [Notice] Ground reflection cutoff skipped ({pe})")

        # ============================================================
        # Stage 3: Multi-Iteration Bilateral MLS Surface Thinning
        # ============================================================
        if self.enable_surface_thinning:
            merged_normal_dedup = self.apply_bilateral_surface_thinning(
                merged_normal_dedup, radius=self.mls_radius
            )
            if merged_colored_dedup:
                merged_colored_dedup.points = merged_normal_dedup.points

        # ============================================================
        # Post-clean ROR (catch MLS-introduced edge artifacts)
        # ============================================================
        post_ror_stats: dict = {}
        if self.enable_sor and 1000 < len(merged_normal_dedup.points) <= 3_000_000:
            print("\n[Post-Clean] Radius Outlier Removal (after MLS)...")
            _, ind = merged_normal_dedup.remove_radius_outlier(
                nb_points=self.ror_min_neighbors, radius=self.ror_radius
            )
            merged_normal_dedup = merged_normal_dedup.select_by_index(ind)
            pts_after = len(merged_normal_dedup.points)
            post_ror_stats = {
                "outliers_removed": pts_before - pts_after,
                "clean_points": pts_after,
            }
            print(f"  ROR: {pts_before:,} -> {pts_after:,} ({pts_before - pts_after:,} removed)")
            if merged_colored_dedup:
                merged_colored_dedup = merged_colored_dedup.select_by_index(ind)

        # ============================================================
        # Post-Clean: Seam Color Blending
        # ============================================================
        if self.enable_color_harmonization and len(merged_normal_dedup.points) <= 3_000_000:
            merged_normal_dedup = self.apply_seam_color_blending(merged_normal_dedup)
        # ============================================================
        # Final benchmark
        # ============================================================
        final_plane_stats = self.compute_plane_residual(
            merged_normal_dedup, threshold=self.plane_threshold
        )
        print(f"\n[Optimized Benchmark] Final dominant plane residual:")
        print(
            f"  Inliers: {final_plane_stats['inliers']:,} ({final_plane_stats['inlier_ratio_pct']}%) "
            f"| Mean={final_plane_stats['mean_residual_mm']}mm | Std={final_plane_stats['std_residual_mm']}mm"
        )

        # ============================================================
        # Save deliverables
        # ============================================================
        print("\n[Save] Writing optimized deliverables...")
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

        # Update transform matrices for each sequence with PGO + leveling
        updated_sequences = {}
        for sid, sdata in submaps.items():
            T_curr = sdata["T"]
            if self.enable_plane_leveling and "T_level" in locals():
                T_final = T_level @ T_curr
            else:
                T_final = T_curr
            sinfo_copy = dict(seq_metadata.get(sid, {}))
            sinfo_copy["scale_to_anchor"] = round(float(sdata["scale"]), 6)
            sinfo_copy["transform_matrix"] = T_final.tolist()
            updated_sequences[sid] = sinfo_copy

        opt_transforms_json = out_dir / "transforms.json"
        if updated_sequences:
            transforms_payload = {
                "method": "Method 2 Multi-View Sim(3) Fusion (Post-Fusion Optimized v2)",
                "anchor_sequence": anchor_id,
                "merge_voxel_size_m": self.voxel_size,
                "total_raw_points": len(merged_normal_full.points),
                "dedup_points": len(merged_normal_dedup.points),
                "sequences": updated_sequences,
            }
            with open(opt_transforms_json, "w", encoding="utf-8") as f:
                json.dump(transforms_payload, f, indent=2, ensure_ascii=False)
            print(f"  Saved Optimized Transforms JSON: {opt_transforms_json}")

        total_time = round(time.time() - t_start, 2)

        report_data = {
            "title": "Post-Fusion 3D Point Cloud Optimization Report (v2)",
            "source_dir": str(f_dir),
            "output_dir": str(out_dir),
            "voxel_size_m": self.voxel_size,
            "total_wall_time_s": total_time,
            "pipeline_order": [
                "1. Global PGO (all-pairs ICP + pose graph optimization)",
                "2. Submap color gain correction",
                "3. Merge + voxel downsample",
                "4. Pre-clean SOR",
                "5. Dominant plane leveling",
                "6. Multi-iteration bilateral MLS thinning",
                "7. Post-clean ROR",
            ],
            "pgo_stats": pgo_stats,
            "color_gain_stats": color_stats,
            "pre_sor_stats": pre_sor_stats,
            "plane_leveling_stats": plane_level_stats,
            "post_ror_stats": post_ror_stats,
            "baseline_metrics": {
                "plane": init_plane_stats,
                "pairwise_precision": init_pairwise,
            },
            "optimized_metrics": {
                "plane": final_plane_stats,
                "pairwise_precision": post_pgo_pairwise,
            },
            "improvements": {
                "plane_mean_residual_reduction_mm": round(
                    init_plane_stats["mean_residual_mm"] - final_plane_stats["mean_residual_mm"], 2
                ),
                "plane_inlier_ratio_gain_pct": round(
                    final_plane_stats["inlier_ratio_pct"] - init_plane_stats["inlier_ratio_pct"], 2
                ),
            },
            "deliverables": {
                "optimized_normal": str(opt_normal_ply),
                "optimized_colored": str(opt_colored_ply) if merged_colored_dedup else None,
                "optimized_normal_full": str(opt_normal_full_ply),
                "optimized_colored_full": (
                    str(opt_colored_full_ply) if len(merged_colored_full.points) > 0 else None
                ),
            },
        }

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(
            f"  Normal PLY:  {opt_normal_ply} "
            f"({opt_normal_ply.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        if merged_colored_dedup:
            print(
                f"  Colored PLY: {opt_colored_ply} "
                f"({opt_colored_ply.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        print(f"  Report:      {report_json}")
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
        "--mls-iterations",
        type=int,
        default=3,
        help="Number of MLS surface thinning iterations (default: 3)",
    )
    parser.add_argument(
        "--mls-radius",
        type=float,
        default=0.05,
        help="MLS search radius in meters (default: 0.05m)",
    )
    parser.add_argument(
        "--submap-icp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable global PGO refinement on aligned submaps",
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
        help="Enable multi-iteration bilateral MLS surface thinning",
    )
    parser.add_argument(
        "--color-harmonize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable submap-aware photometric color gain correction",
    )
    parser.add_argument(
        "--sor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dual-stage denoising (SOR pre-clean + ROR post-clean)",
    )
    args = parser.parse_args()

    optimizer = PostFusionOptimizer(
        voxel_size=args.voxel_size,
        mls_iterations=args.mls_iterations,
        mls_radius=args.mls_radius,
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
