#!/usr/bin/env python3
"""
General-Purpose Multi-Sequence Point Cloud Fusion using Method 2
(LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + Coarse-to-Fine VGICP + Sim(3) Global Pose Graph Optimization)

Features:
  1. Arbitrary N sequences: accepts any number of sequence directories.
  2. Automatic Graph Topology & Anchor Selection:
     - Global image retrieval (DINO-SALAD) to find cross-sequence candidate overlaps.
     - 2D matching (ALIKED + LightGlue) & 3D lifting to solve Sim(3) for valid edges.
     - Maximum Spanning Tree (MST) and auto-selection of the optimal anchor reference.
  3. Global Sim(3) Pose Graph Optimization (PGO):
     - Jointly distributes loop closure residuals across scale, rotation, and translation.
     - Closed-form weighted linear least squares for optimal scales.
     - Geodesic SO(3) Lie algebra Gauss-Newton optimization for rotations.
     - Closed-form weighted linear least squares for translations.
  4. Coarse-to-Fine Multi-Scale VGICP Refinement with Safety Checks:
     - Multi-scale voxel matching with normal estimation.
     - Dual-threshold safety gate (fitness retention and inlier RMSE).
  5. Multi-View Statistical Outlier Removal (SOR) & Quality Metrics:
     - Surface-level point cloud denoising on merged maps.
     - Pairwise overlap quantitative metrics (mean distance, precision @ 2cm / 5cm).
  6. Unified Multi-Camera Trajectory & BEV Export:
     - Transforms all camera poses into anchor coordinates.
     - Multi-sequence color-coded BEV trajectory map with legend.
  7. Dual-Version PLY Export:
     - Normal true-color fusion (full and voxel-deduplicated).
     - Distinct-color fusion (scalable golden-ratio HSV palette for any N).
     - Complete transform JSON and metrics report.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import small_gicp
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R_scipy

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abot_recon.sparse_loop.retrieval import RetrievalConfig, compute_descriptors
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
from scripts.align_method2_lightglue_umeyama import ransac_umeyama


def generate_distinct_colors(num_colors: int) -> List[Tuple[float, float, float]]:
    """Generate `num_colors` maximally distinct RGB colors using Golden Ratio HSV."""
    fixed_palette = [
        (0.92, 0.20, 0.20),  # Red
        (0.15, 0.80, 0.25),  # Green
        (0.12, 0.56, 1.00),  # Blue
        (1.00, 0.80, 0.08),  # Gold/Yellow
        (0.85, 0.20, 0.85),  # Magenta
        (0.10, 0.85, 0.85),  # Cyan
        (1.00, 0.50, 0.10),  # Orange
        (0.55, 0.25, 0.80),  # Purple
        (0.60, 0.90, 0.20),  # Lime
        (0.90, 0.35, 0.55),  # Pink
    ]
    if num_colors <= len(fixed_palette):
        return fixed_palette[:num_colors]

    colors = []
    golden_ratio = 0.618033988749895
    h = 0.1
    for _ in range(num_colors):
        h = (h + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((round(r, 4), round(g, 4), round(b, 4)))
    return colors


def write_multi_sequence_bev(
    path: Path,
    camera_trajectories: Dict[str, np.ndarray],
    palette: Dict[str, Tuple[float, float, float]],
    point_counts: Dict[str, int],
    size: int = 1600,
    plane: str = "auto",
) -> str:
    """Export unified multi-sequence camera trajectory BEV visualization."""
    if not camera_trajectories:
        return "none"

    all_poses = np.concatenate(list(camera_trajectories.values()), axis=0)
    centers_3d = all_poses[:, :3, 3]

    if plane.lower() == "auto":
        span_3d = centers_3d.max(axis=0) - centers_3d.min(axis=0)
        plane = "xz" if span_3d[1] <= min(span_3d[0], span_3d[2]) else "xy"

    axes = (0, 2) if plane.lower() == "xz" else (0, 1)
    all_2d = centers_3d[:, axes]

    lower = all_2d.min(axis=0)
    upper = all_2d.max(axis=0)
    span = np.maximum(upper - lower, 1e-4)
    margin = max(60, size // 16)
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])

    image = Image.new("RGB", (size, size), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    grid_color = (222, 226, 232)

    # Grid lines & border
    for fraction in np.linspace(0.0, 1.0, 11):
        coordinate = int(round(margin + fraction * (size - 2 * margin)))
        draw.line((coordinate, margin, coordinate, size - margin), fill=grid_color, width=1)
        draw.line((margin, coordinate, size - margin, coordinate), fill=grid_color, width=1)
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(150, 158, 170), width=2)

    line_width = max(3, size // 350)
    marker_radius = max(6, size // 130)

    # Draw trajectories
    for sid, poses in camera_trajectories.items():
        traj_3d = poses[:, :3, 3]
        pts_2d = traj_3d[:, axes]
        offset_x = margin + (size - 2 * margin - span[0] * scale) / 2
        offset_y = margin + (size - 2 * margin - span[1] * scale) / 2
        canvas = np.empty((len(pts_2d), 2), dtype=np.float64)
        canvas[:, 0] = offset_x + (pts_2d[:, 0] - lower[0]) * scale
        canvas[:, 1] = size - offset_y - (pts_2d[:, 1] - lower[1]) * scale

        rgb_tuple = tuple(int(round(c * 255)) for c in palette[sid])

        # Draw trajectory polyline
        for idx in range(len(canvas) - 1):
            draw.line((*canvas[idx], *canvas[idx + 1]), fill=rgb_tuple, width=line_width)

        # Draw start marker
        sx, sy = canvas[0]
        draw.ellipse((sx - marker_radius, sy - marker_radius, sx + marker_radius, sy + marker_radius),
                     fill=(35, 180, 50), outline=(255, 255, 255), width=2)

        # Draw end marker
        ex, ey = canvas[-1]
        draw.ellipse((ex - marker_radius, ey - marker_radius, ex + marker_radius, ey + marker_radius),
                     fill=(230, 45, 45), outline=(255, 255, 255), width=2)

    # Title & Subtitle
    draw.text((margin, 16), f"Unified Multi-Stream BEV Trajectory ({plane.upper()} plane)", fill=(25, 30, 38))
    draw.text((margin, size - margin + 12), f"{len(camera_trajectories)} streams | {len(all_poses)} total poses | span: {span[0]:.2f}m x {span[1]:.2f}m", fill=(75, 82, 94))

    # Legend in top right
    leg_x = size - margin - 280
    leg_y = margin + 12
    draw.rectangle((leg_x - 10, leg_y - 6, size - margin - 6, leg_y + 38 + len(camera_trajectories) * 24),
                   fill=(255, 255, 255), outline=(180, 185, 195), width=1)
    draw.text((leg_x, leg_y), "Camera Sequences:", fill=(30, 35, 45))
    leg_y += 20
    for sid in sorted(camera_trajectories.keys()):
        rgb_tuple = tuple(int(round(c * 255)) for c in palette[sid])
        draw.rectangle((leg_x, leg_y + 2, leg_x + 16, leg_y + 14), fill=rgb_tuple, outline=(50, 50, 50))
        pts_count = point_counts.get(sid, 0)
        poses_count = len(camera_trajectories[sid])
        draw.text((leg_x + 22, leg_y), f"{sid}: {poses_count}f, {pts_count/1e6:.1f}M pts", fill=(40, 45, 55))
        leg_y += 22
    # Start / End markers indicator
    draw.ellipse((leg_x, leg_y + 4, leg_x + 10, leg_y + 14), fill=(35, 180, 50), outline=(50, 50, 50))
    draw.text((leg_x + 14, leg_y + 2), "Start", fill=(60, 65, 75))
    draw.ellipse((leg_x + 65, leg_y + 4, leg_x + 75, leg_y + 14), fill=(230, 45, 45), outline=(50, 50, 50))
    draw.text((leg_x + 79, leg_y + 2), "End", fill=(60, 65, 75))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return plane


def optimize_sim3_pose_graph(
    edges: List[EdgeResult],
    initial_transforms: Dict[str, Tuple[float, np.ndarray]],
    anchor_id: str,
    max_iters: int = 20,
) -> Tuple[Dict[str, Tuple[float, np.ndarray]], dict]:
    """
    Global Sim(3) Pose Graph Optimization (PGO) over all registered loop closures.
    Jointly distributes residual loop closure errors across scales, rotations, and translations.
    """
    node_ids = list(initial_transforms.keys())
    if len(node_ids) <= 2 or len(edges) < len(node_ids):
        return initial_transforms, {"optimized": False, "reason": "insufficient_edges"}

    var_nodes = [n for n in node_ids if n != anchor_id]
    node_to_idx = {n: i for i, n in enumerate(var_nodes)}

    # High quality valid edges for PGO
    valid_edges = [
        e for e in edges
        if e.inliers >= 15 and e.rmse < 0.06 and 0.5 <= e.scale <= 2.0
    ]

    if len(valid_edges) < len(node_ids):
        return initial_transforms, {"optimized": False, "reason": "no_closed_loops"}

    # Initial residual calculation
    def compute_graph_residuals(curr_transforms):
        res_scale, res_rot, res_trans = [], [], []
        for e in valid_edges:
            su, Tu = curr_transforms[e.src]
            sv, Tv = curr_transforms[e.tgt]

            # Scale residual: ln(su / sv) - ln(s_edge)
            res_scale.append(abs(np.log(su / sv) - np.log(e.scale)))

            # Rotation residual: R_diff = (Rv @ R_edge)^T @ Ru
            R_pred = Tv[:3, :3] @ e.R
            R_diff = R_pred.T @ Tu[:3, :3]
            angle = float(np.linalg.norm(R_scipy.from_matrix(R_diff).as_rotvec()))
            res_rot.append(angle)

            # Translation residual: (tu - tv) - sv * Rv @ t_edge
            t_pred = sv * (Tv[:3, :3] @ e.t)
            t_diff = (Tu[:3, 3] - Tv[:3, 3]) - t_pred
            res_trans.append(float(np.linalg.norm(t_diff)))
        return float(np.mean(res_scale)), float(np.mean(res_rot)), float(np.mean(res_trans))

    init_res_s, init_res_r, init_res_t = compute_graph_residuals(initial_transforms)

    # 1. Scale Optimization (Linear Least Squares on log scale)
    A_scale, b_scale = [], []
    for e in valid_edges:
        u, v = e.src, e.tgt
        w = np.sqrt(float(e.inliers) / (e.rmse + 0.01))
        row = np.zeros(len(var_nodes))
        if u in node_to_idx:
            row[node_to_idx[u]] += 1.0
        if v in node_to_idx:
            row[node_to_idx[v]] -= 1.0
        A_scale.append(row * w)
        b_scale.append(np.log(e.scale) * w)

    sol_scale, _, _, _ = np.linalg.lstsq(np.array(A_scale), np.array(b_scale), rcond=None)
    opt_scales = {anchor_id: 1.0}
    for n in var_nodes:
        opt_scales[n] = float(np.exp(sol_scale[node_to_idx[n]]))

    # 2. Rotation Optimization (SO(3) Lie Algebra Gauss-Newton)
    opt_R = {n: initial_transforms[n][1][:3, :3].copy() for n in node_ids}
    opt_R[anchor_id] = np.eye(3, dtype=np.float64)

    for _ in range(max_iters):
        J_list, r_list = [], []
        for e in valid_edges:
            u, v = e.src, e.tgt
            w = np.sqrt(float(e.inliers) / (e.rmse + 0.01))
            R_pred = opt_R[v] @ e.R
            R_diff = R_pred.T @ opt_R[u]
            r = R_scipy.from_matrix(R_diff).as_rotvec()

            J_row = np.zeros((3, 3 * len(var_nodes)))
            if u in node_to_idx:
                J_row[:, 3 * node_to_idx[u] : 3 * node_to_idx[u] + 3] += np.eye(3)
            if v in node_to_idx:
                J_row[:, 3 * node_to_idx[v] : 3 * node_to_idx[v] + 3] -= R_diff
            J_list.append(J_row * w)
            r_list.append(r * w)

        J = np.vstack(J_list)
        res = np.concatenate(r_list)
        delta, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
        for n in var_nodes:
            d_omega = delta[3 * node_to_idx[n] : 3 * node_to_idx[n] + 3]
            opt_R[n] = opt_R[n] @ R_scipy.from_rotvec(d_omega).as_matrix()
        if np.linalg.norm(delta) < 1e-5:
            break

    # 3. Translation Optimization (Linear Least Squares)
    C_trans, d_trans = [], []
    for e in valid_edges:
        u, v = e.src, e.tgt
        w = np.sqrt(float(e.inliers) / (e.rmse + 0.01))
        rhs = opt_scales[v] * (opt_R[v] @ e.t)
        for axis in range(3):
            row = np.zeros(3 * len(var_nodes))
            if u in node_to_idx:
                row[3 * node_to_idx[u] + axis] += 1.0
            if v in node_to_idx:
                row[3 * node_to_idx[v] + axis] -= 1.0
            C_trans.append(row * w)
            d_trans.append(rhs[axis] * w)

    sol_trans, _, _, _ = np.linalg.lstsq(np.array(C_trans), np.array(d_trans), rcond=None)
    opt_t = {anchor_id: np.zeros(3, dtype=np.float64)}
    for n in var_nodes:
        opt_t[n] = sol_trans[3 * node_to_idx[n] : 3 * node_to_idx[n] + 3]

    refined_transforms = {}
    for n in node_ids:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = opt_R[n]
        T[:3, 3] = opt_t[n]
        refined_transforms[n] = (opt_scales[n], T)

    post_res_s, post_res_r, post_res_t = compute_graph_residuals(refined_transforms)

    # Acceptance check: total weighted residual must decrease or remain tight
    pgo_improved = (post_res_r <= init_res_r * 1.05 and post_res_t <= init_res_t * 1.05)
    final_transforms = refined_transforms if pgo_improved else initial_transforms

    info = {
        "optimized": pgo_improved,
        "valid_edges_count": len(valid_edges),
        "initial_residual": {
            "scale_error": round(init_res_s, 6),
            "rot_error_deg": round(float(np.rad2deg(init_res_r)), 3),
            "trans_error_mm": round(init_res_t * 1000, 2),
        },
        "optimized_residual": {
            "scale_error": round(post_res_s, 6),
            "rot_error_deg": round(float(np.rad2deg(post_res_r)), 3),
            "trans_error_mm": round(post_res_t * 1000, 2),
        },
    }
    return final_transforms, info


def evaluate_pairwise_overlap_metrics(
    aligned_pcds: Dict[str, o3d.geometry.PointCloud],
    voxel_size: float = 0.03,
    max_dist: float = 0.05,
) -> Dict[str, dict]:
    """Compute quantitative map-to-map consistency across overlapping pairs."""
    results = {}
    seq_ids = list(aligned_pcds.keys())
    for i in range(len(seq_ids)):
        for j in range(i + 1, len(seq_ids)):
            si, sj = seq_ids[i], seq_ids[j]
            pcd_i = aligned_pcds[si].voxel_down_sample(voxel_size)
            pcd_j = aligned_pcds[sj].voxel_down_sample(voxel_size)
            if len(pcd_i.points) == 0 or len(pcd_j.points) == 0:
                continue

            bb_i = pcd_i.get_axis_aligned_bounding_box()
            bb_j = pcd_j.get_axis_aligned_bounding_box()
            min_overlap = np.maximum(bb_i.min_bound, bb_j.min_bound)
            max_overlap = np.minimum(bb_i.max_bound, bb_j.max_bound)
            if np.any(min_overlap >= max_overlap):
                continue

            dists_ij = np.asarray(pcd_i.compute_point_cloud_distance(pcd_j))
            dists_ji = np.asarray(pcd_j.compute_point_cloud_distance(pcd_i))

            in_zone_ij = dists_ij[dists_ij < max_dist * 2]
            in_zone_ji = dists_ji[dists_ji < max_dist * 2]

            if len(in_zone_ij) > 50 and len(in_zone_ji) > 50:
                mean_dist = float(0.5 * (np.mean(in_zone_ij) + np.mean(in_zone_ji)))
                prec_2cm = float(0.5 * (np.mean(in_zone_ij < 0.02) + np.mean(in_zone_ji < 0.02)) * 100.0)
                prec_5cm = float(0.5 * (np.mean(in_zone_ij < 0.05) + np.mean(in_zone_ji < 0.05)) * 100.0)
                results[f"{si}<->{sj}"] = {
                    "mean_distance_mm": round(mean_dist * 1000, 2),
                    "precision_at_2cm_pct": round(prec_2cm, 2),
                    "precision_at_5cm_pct": round(prec_5cm, 2),
                    "overlap_points_sampled": int(len(in_zone_ij) + len(in_zone_ji)),
                }
    return results


@dataclass
class SequenceData:
    id: str
    dir_path: Path
    pcd_raw: o3d.geometry.PointCloud
    pcd_down: o3d.geometry.PointCloud
    colors_tensor: torch.Tensor
    world_tensor: torch.Tensor
    conf_tensor: torch.Tensor
    camera_poses: np.ndarray
    keyframe_indices: np.ndarray
    descriptors: np.ndarray


@dataclass
class EdgeResult:
    src: str
    tgt: str
    fs: int
    ft: int
    matches: int
    inliers: int
    inlier_ratio: float
    scale: float
    R: np.ndarray
    t: np.ndarray
    rmse: float
    T_fine: np.ndarray
    score: float


class Method2MultiViewFusion:
    """General-purpose automated multi-sequence point cloud fusion engine."""

    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        keyframe_stride: int = 5,
        top_k_pairs: int = 8,
        ransac_thresh: float = 0.08,
        merge_voxel_size: float = 0.015,
        downsample_voxel_size: float = 0.03,
        enable_pgo: bool = True,
        enable_sor: bool = True,
        multi_scale_icp: bool = True,
        bev_size: int = 1600,
    ):
        self.device = torch.device(device)
        self.keyframe_stride = max(1, keyframe_stride)
        self.top_k_pairs = max(1, top_k_pairs)
        self.ransac_thresh = ransac_thresh
        self.merge_voxel_size = merge_voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.enable_pgo = enable_pgo
        self.enable_sor = enable_sor
        self.multi_scale_icp = multi_scale_icp
        self.bev_size = bev_size

        print(f"[Engine] Initializing models on {self.device}...")
        self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)
        self.retrieval_cfg = RetrievalConfig(
            salad_checkpoint=REPO_ROOT / "checkpoints/loop/dino_salad.ckpt",
            dino_checkpoint=REPO_ROOT / "checkpoints/loop/dinov2_vitb14_pretrain.pth",
            backbone="dinov2_vitb14",
            verbose=False,
        )

    def load_sequences(self, dirs: List[Path | str]) -> Dict[str, SequenceData]:
        """Load point clouds, tensors, and compute global keyframe descriptors."""
        sequences: Dict[str, SequenceData] = {}
        for d in dirs:
            dir_path = Path(d).resolve()
            sid = dir_path.name
            if sid.startswith("data_") and sid.endswith("_loop"):
                sid = sid[5:-5]
            elif sid.endswith("_loop"):
                sid = sid[:-5]

            print(f"  Loading sequence [{sid}] from {dir_path.name}...")
            ply_file = dir_path / "reconstruction.ply"
            if not ply_file.is_file():
                raise FileNotFoundError(f"Missing reconstruction.ply in {dir_path}")

            colors = torch.load(dir_path / "colors.pt", map_location="cpu", weights_only=True)
            world = torch.load(dir_path / "world_points.pt", map_location="cpu", weights_only=True)
            conf = torch.load(dir_path / "confidence.pt", map_location="cpu", weights_only=True)
            pcd = o3d.io.read_point_cloud(str(ply_file))
            pcd_down = pcd.voxel_down_sample(self.downsample_voxel_size)
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))

            pose_file = dir_path / "camera_poses.npy"
            if not pose_file.is_file():
                raise FileNotFoundError(f"Missing camera_poses.npy in {dir_path}")
            camera_poses = np.load(pose_file)
            kf_idx = np.arange(0, len(colors), self.keyframe_stride)
            desc = compute_descriptors(colors[kf_idx].numpy(), self.retrieval_cfg, self.device)
            sequences[sid] = SequenceData(
                id=sid,
                dir_path=dir_path,
                pcd_raw=pcd,
                pcd_down=pcd_down,
                colors_tensor=colors,
                world_tensor=world,
                conf_tensor=conf,
                camera_poses=camera_poses,
                keyframe_indices=kf_idx,
                descriptors=desc,
            )
            print(f"    -> {len(pcd.points):,} raw points, {len(kf_idx)} keyframes indexed.")
        return sequences

    @staticmethod
    def _trajectory_prior(seq_s: SequenceData, seq_t: SequenceData) -> Optional[dict]:
        """Estimate a map Sim(3) from camera-center trajectories."""
        def sample(seq: SequenceData) -> np.ndarray:
            p = np.asarray(seq.camera_poses[:, :3, 3], dtype=np.float64)
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            if len(p) < 8 or d[-1] < 0.5:
                return np.empty((0, 3))
            u = np.linspace(0.0, d[-1], 40)
            return np.stack([np.interp(u, d, p[:, k]) for k in range(3)], axis=1)

        src = sample(seq_s)
        tgt = sample(seq_t)
        if len(src) == 0 or len(tgt) == 0:
            return None
        candidates = []
        for reverse in (False, True):
            dst = tgt[::-1] if reverse else tgt
            s, R, t = ransac_umeyama.__globals__["umeyama_svd"](src, dst, estimate_scale=True)
            err = np.linalg.norm(s * (src @ R.T) + t - dst, axis=1)
            candidates.append((float(np.sqrt(np.mean(err * err))), reverse, s, R, t))
        rmse, reverse, s, R, t = min(candidates, key=lambda x: x[0])
        if not (0.5 <= s <= 1.5) or rmse > 0.15:
            return None
        return {"scale": s, "R": R, "t": t, "rmse": rmse, "reverse": reverse}

    def match_pair(self, seq_s: SequenceData, seq_t: SequenceData) -> Optional[EdgeResult]:
        """Match 2D features between two sequences and solve Umeyama Sim(3) + coarse-to-fine VGICP."""
        sim = seq_s.descriptors @ seq_t.descriptors.T
        max_sim = float(sim.max())
        if max_sim < 0.25:  # Skip completely disjoint pairs early
            return None

        top_flat = np.argsort(-sim, axis=None)[: self.top_k_pairs * 4]
        candidate_pairs = []
        seen_source = set()
        seen_target = set()
        for flat in top_flat:
            i, j = np.unravel_index(flat, sim.shape)
            fs, ft = int(seq_s.keyframe_indices[i]), int(seq_t.keyframe_indices[j])
            if fs in seen_source or ft in seen_target:
                continue
            if len(candidate_pairs) >= self.top_k_pairs:
                break
            candidate_pairs.append((fs, ft, float(sim[i, j])))
            seen_source.add(fs)
            seen_target.add(ft)

        pair_solutions = []
        all_p_s, all_p_t = [], []
        candidate_models = []

        for fs, ft, score in candidate_pairs:
            img_s = seq_s.colors_tensor[fs].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            img_t = seq_t.colors_tensor[ft].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

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
            h_s, w_s = seq_s.colors_tensor.shape[1:3]
            h_t, w_t = seq_t.colors_tensor.shape[1:3]
            for (xa, ya), (xb, yb) in zip(kpts_s, kpts_t):
                ia_y, ia_x = min(max(int(round(ya)), 0), h_s - 1), min(max(int(round(xa)), 0), w_s - 1)
                ib_y, ib_x = min(max(int(round(yb)), 0), h_t - 1), min(max(int(round(xb)), 0), w_t - 1)
                if seq_s.conf_tensor[fs, ia_y, ia_x] > 0.05 and seq_t.conf_tensor[ft, ib_y, ib_x] > 0.05:
                    pa = seq_s.world_tensor[fs, ia_y, ia_x].numpy()
                    pb = seq_t.world_tensor[ft, ib_y, ib_x].numpy()
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        p_s.append(pa)
                        p_t.append(pb)

            if len(p_s) < 4:
                continue
            p_s, p_t = np.asarray(p_s), np.asarray(p_t)
            res = ransac_umeyama(p_s, p_t, estimate_scale=True, iters=3000, thresh=self.ransac_thresh)
            if res is None:
                continue
            s, R, t_vec, inliers, rmse = res
            pair_solutions.append({
                "fs": fs, "ft": ft, "matches": len(matches), "pairs": len(p_s),
                "inliers": len(inliers), "inlier_ratio": len(inliers) / len(p_s),
                "scale": s, "R": R, "t": t_vec, "rmse": rmse,
            })
            candidate_models.append((p_s, p_t, pair_solutions[-1]))
            print(
                f"    [Candidate {seq_s.id}->{seq_t.id}] frames=({fs},{ft}) "
                f"pairs={len(p_s)} inliers={len(inliers)} "
                f"ratio={len(inliers)/len(p_s):.3f} scale={s:.4f} rmse={rmse*1000:.1f}mm"
            )
            all_p_s.append(p_s)
            all_p_t.append(p_t)

        if not pair_solutions:
            return None

        pair_solutions.sort(key=lambda x: (-x["inliers"], -x["inlier_ratio"], x["rmse"]))

        def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
            c = np.clip((np.trace(a.T @ b) - 1.0) * 0.5, -1.0, 1.0)
            return float(np.arccos(c))

        # Consistency clustering over candidate Sim(3) models
        clusters = []
        for model in pair_solutions:
            assigned = False
            for cluster in clusters:
                ref = cluster[0]
                scale_delta = abs(model["scale"] / ref["scale"] - 1.0)
                angle_delta = rotation_distance(model["R"], ref["R"])
                translation_delta = float(np.linalg.norm(model["t"] - ref["t"]))
                if scale_delta < 0.08 and angle_delta < np.deg2rad(15.0) and translation_delta < 0.30:
                    cluster.append(model)
                    assigned = True
                    break
            if not assigned:
                clusters.append([model])
        clusters.sort(key=lambda c: (-sum(x["inliers"] for x in c), -len(c), min(x["rmse"] for x in c)))
        if not clusters:
            return None
        best_cluster = clusters[0]
        best = max(best_cluster, key=lambda x: (x["inliers"], x["inlier_ratio"], -x["rmse"]))
        best["consistent_candidates"] = len(best_cluster)
        print(f"  [Model consensus] {len(best_cluster)}/{len(pair_solutions)} candidate transforms agree")

        # Refine from correspondences belonging only to this transform cluster
        agreeing_points = [(ps, pt) for ps, pt, model in candidate_models if model in best_cluster]
        if len(agreeing_points) >= 2:
            refined = ransac_umeyama(
                np.concatenate([x[0] for x in agreeing_points]),
                np.concatenate([x[1] for x in agreeing_points]),
                estimate_scale=True, iters=5000, thresh=self.ransac_thresh,
            )
            if refined is not None:
                s, R, t_vec, inliers, rmse = refined
                best.update({"scale": s, "R": R, "t": t_vec,
                             "inliers": int(len(inliers)),
                             "inlier_ratio": float(len(inliers) / sum(len(x[0]) for x in agreeing_points)),
                             "rmse": float(rmse)})

        # Coarse-to-fine VGICP refinement with small_gicp
        pts_coarse = (np.asarray(seq_s.pcd_down.points, dtype=np.float64) * best["scale"]) @ best["R"].T + best["t"]
        pts_target = np.asarray(seq_t.pcd_down.points, dtype=np.float64)
        T_delta_accum = np.eye(4, dtype=np.float64)
        pts_current = pts_coarse.copy()

        stages = [(0.08, 0.08, 30), (0.04, 0.04, 25)] if self.multi_scale_icp else [(0.08, 0.08, 40)]
        for vox_res, max_corr, iters in stages:
            gicp_stage = small_gicp.align(
                target_points=pts_target,
                source_points=pts_current,
                init_T_target_source=np.eye(4, dtype=np.float64),
                registration_type="VGICP",
                voxel_resolution=vox_res,
                downsampling_resolution=self.downsample_voxel_size,
                max_correspondence_distance=max_corr,
                max_iterations=iters,
                num_threads=8,
            )
            if gicp_stage.converged:
                T_delta_accum = gicp_stage.T_target_source @ T_delta_accum
                pts_current = (pts_current @ gicp_stage.T_target_source[:3, :3].T) + gicp_stage.T_target_source[:3, 3]

        # Edge score combines inliers and low RMSE
        edge_score = float(best["inliers"]) * (1.0 / (best["rmse"] + 0.01))

        # Overlap safeguard: VGICP must not decrease actual overlap
        coarse_eval = o3d.geometry.PointCloud()
        coarse_eval.points = o3d.utility.Vector3dVector(pts_coarse)
        fine_eval = o3d.geometry.PointCloud()
        fine_eval.points = o3d.utility.Vector3dVector(pts_current)
        target_eval = seq_t.pcd_down

        coarse_metric = o3d.pipelines.registration.evaluate_registration(coarse_eval, target_eval, 0.06)
        fine_metric = o3d.pipelines.registration.evaluate_registration(fine_eval, target_eval, 0.06)

        accept_delta = (fine_metric.fitness >= coarse_metric.fitness * 0.95) and (fine_metric.inlier_rmse <= coarse_metric.inlier_rmse * 1.10)
        T_delta = T_delta_accum if accept_delta else np.eye(4, dtype=np.float64)
        if not accept_delta:
            print("  [VGICP] rejected: overlap decreased or RMSE increased")

        T_rigid = np.eye(4, dtype=np.float64)
        T_rigid[:3, :3] = best["R"]
        T_rigid[:3, 3] = best["t"]
        T_fine = T_delta @ T_rigid

        return EdgeResult(
            src=seq_s.id, tgt=seq_t.id, fs=best["fs"], ft=best["ft"],
            matches=best["matches"], inliers=best["inliers"],
            inlier_ratio=best["inlier_ratio"], scale=best["scale"],
            R=best["R"], t=best["t"], rmse=best["rmse"],
            T_fine=T_fine, score=edge_score,
        )

    def build_spanning_tree(
        self,
        sequences: Dict[str, SequenceData],
        anchor_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Tuple[float, np.ndarray]], List[EdgeResult], List[EdgeResult]]:
        """Build the registration graph and propagate initial MST transforms."""
        seq_ids = list(sequences.keys())
        print(f"\n[Graph] Evaluating pairwise connectivity across {len(seq_ids)} sequences...")

        edges: List[EdgeResult] = []
        for i in range(len(seq_ids)):
            for j in range(len(seq_ids)):
                if i != j:
                    sid_s, sid_t = seq_ids[i], seq_ids[j]
                    res = self.match_pair(sequences[sid_s], sequences[sid_t])
                    if res is not None:
                        edges.append(res)
                        print(
                            f"  Edge [{sid_s} -> {sid_t}]: {res.inliers} inliers, "
                            f"scale={res.scale:.4f}, RMSE={res.rmse*1000:.1f}mm (score={res.score:.1f})"
                        )

        if not edges:
            raise RuntimeError("No overlapping sequences could be registered! Check inputs or keyframe stride.")

        # Determine Anchor
        if anchor_id is not None:
            if anchor_id not in sequences:
                raise ValueError(f"Specified anchor '{anchor_id}' not found in sequences {seq_ids}")
            root = anchor_id
        else:
            # Score each sequence by total connectivity in graph
            node_scores: Dict[str, float] = {sid: 0.0 for sid in seq_ids}
            for e in edges:
                node_scores[e.tgt] += e.score
                node_scores[e.src] += e.score * 0.5
            root = max(node_scores, key=lambda k: node_scores[k])
        print(f"\n[Anchor] Selected sequence [{root}] as Global Reference Anchor.")

        # Build adjacency graph for Maximum Spanning Tree
        adj: Dict[str, List[Tuple[str, EdgeResult]]] = {sid: [] for sid in seq_ids}
        for e in edges:
            adj[e.tgt].append((e.src, e))

        transforms_to_anchor: Dict[str, Tuple[float, np.ndarray]] = {
            root: (1.0, np.eye(4, dtype=np.float64))
        }

        # Greedy tree expansion (Prim-like)
        visited = {root}
        mst_edges: List[EdgeResult] = []

        while len(visited) < len(seq_ids):
            best_candidate = None
            best_edge = None
            best_parent = None
            best_weight = -1.0

            for u in visited:
                for v, edge in adj[u]:
                    if v not in visited:
                        if edge.score > best_weight:
                            best_weight = edge.score
                            best_candidate = v
                            best_edge = edge
                            best_parent = u

            if best_candidate is None:
                unvisited = [s for s in seq_ids if s not in visited]
                print(f"  [Warning] Graph disconnected! Remaining sequences cannot be aligned: {unvisited}")
                break

            parent_scale, parent_T = transforms_to_anchor[best_parent]
            edge_scale = best_edge.scale
            edge_T = best_edge.T_fine

            combined_scale = parent_scale * edge_scale
            T_scaled = edge_T.copy()
            T_scaled[:3, 3] *= parent_scale
            combined_T = parent_T @ T_scaled

            transforms_to_anchor[best_candidate] = (combined_scale, combined_T)
            visited.add(best_candidate)
            mst_edges.append(best_edge)
            print(f"  [Tree] Connected [{best_candidate}] -> [{best_parent}] (cum_scale = {combined_scale:.4f})")

        return root, transforms_to_anchor, mst_edges, edges

    def fuse(
        self,
        recon_dirs: List[Path | str],
        output_dir: Path | str,
        anchor_id: Optional[str] = None,
    ) -> dict:
        """Run complete end-to-end Method 2 fusion on all sequences."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        print("=" * 80)
        print("General-Purpose Method 2 Multi-Sequence Point Cloud Fusion Engine")
        print("=" * 80)
        print(f"Output Directory:    {out_path}")
        print(f"Merge Voxel Size:    {self.merge_voxel_size}m")
        print(f"Pose Graph Opt (PGO):{self.enable_pgo}")
        print(f"SOR Denoising:       {self.enable_sor}")
        print(f"Multi-Scale VGICP:   {self.multi_scale_icp}")
        print(f"Sequences Count:     {len(recon_dirs)}\n")

        # 1. Load data
        print("[Step 1] Loading sequences & computing descriptors...")
        sequences = self.load_sequences(recon_dirs)

        # 2. Build Spanning Tree & solve initial transforms to anchor
        print("\n[Step 2] Building topology graph & solving Sim(3) transforms...")
        anchor, transforms, mst_edges, all_edges = self.build_spanning_tree(sequences, anchor_id=anchor_id)

        # 2.1 Global Pose Graph Optimization (PGO) over all loop closures
        pgo_info = {"optimized": False}
        if self.enable_pgo:
            print("\n[Step 2.1] Running Sim(3) Global Pose Graph Optimization (PGO)...")
            transforms, pgo_info = optimize_sim3_pose_graph(
                edges=all_edges,
                initial_transforms=transforms,
                anchor_id=anchor,
            )
            if pgo_info.get("optimized", False):
                init_err = pgo_info["initial_residual"]
                opt_err = pgo_info["optimized_residual"]
                print(
                    f"  [PGO] Success! Loop rotation error: {init_err['rot_error_deg']:.2f}° -> {opt_err['rot_error_deg']:.2f}°, "
                    f"translation error: {init_err['trans_error_mm']:.1f}mm -> {opt_err['trans_error_mm']:.1f}mm"
                )
            else:
                print(f"  [PGO] Retained tree solution ({pgo_info.get('reason', 'no error reduction')}).")

        # 3. Generate colors palette
        colors = generate_distinct_colors(len(sequences))
        palette = {sid: np.array(col, dtype=np.float64) for sid, col in zip(sequences.keys(), colors)}

        # 4. Transform raw point clouds & camera trajectories
        print("\n[Step 3] Applying transformations to point clouds & trajectories...")
        aligned_normal = []
        aligned_colored = []
        aligned_pcds_map = {}
        transformed_trajectories = {}
        point_stats = {}

        for sid, seq in sequences.items():
            if sid not in transforms:
                continue
            scale, T = transforms[sid]
            raw = seq.pcd_raw
            pts_scaled = np.asarray(raw.points, dtype=np.float64) * scale
            pts_trans = (pts_scaled @ T[:3, :3].T) + T[:3, 3]

            # Normal
            pcd_n = o3d.geometry.PointCloud()
            pcd_n.points = o3d.utility.Vector3dVector(pts_trans)
            pcd_n.colors = raw.colors
            aligned_normal.append(pcd_n)
            aligned_pcds_map[sid] = pcd_n

            # Colored
            pcd_c = o3d.geometry.PointCloud()
            pcd_c.points = o3d.utility.Vector3dVector(pts_trans)
            col_matrix = np.tile(palette[sid], (len(pts_trans), 1))
            pcd_c.colors = o3d.utility.Vector3dVector(col_matrix)
            aligned_colored.append(pcd_c)

            # Transform camera trajectory
            poses = seq.camera_poses.copy()
            centers_scaled = poses[:, :3, 3] * scale
            poses_trans = poses.copy()
            poses_trans[:, :3, 3] = (centers_scaled @ T[:3, :3].T) + T[:3, 3]
            poses_trans[:, :3, :3] = T[:3, :3] @ poses[:, :3, :3]
            transformed_trajectories[sid] = poses_trans

            point_stats[sid] = {
                "raw_points": len(pts_trans),
                "scale_to_anchor": round(float(scale), 6),
                "color_rgb": palette[sid].tolist(),
                "transform_matrix": T.tolist(),
                "camera_poses_count": len(poses),
            }
            print(f"  Seq [{sid}]: {len(pts_trans):,} points transformed (scale s={scale:.4f})")

        # 4.1 Quantitative pairwise overlap evaluation
        print("\n[Step 3.1] Evaluating quantitative overlap consistency metrics...")
        overlap_metrics = evaluate_pairwise_overlap_metrics(
            aligned_pcds=aligned_pcds_map,
            voxel_size=self.downsample_voxel_size,
        )
        for pair_name, m_info in overlap_metrics.items():
            print(
                f"  Pair [{pair_name}]: Mean Distance = {m_info['mean_distance_mm']:.1f}mm | "
                f"Precision @ 2cm = {m_info['precision_at_2cm_pct']:.1f}% | "
                f"Precision @ 5cm = {m_info['precision_at_5cm_pct']:.1f}%"
            )

        # 5. Merge point clouds
        print("\n[Step 4] Merging raw point clouds...")
        full_normal = o3d.geometry.PointCloud()
        full_colored = o3d.geometry.PointCloud()
        for pn, pc in zip(aligned_normal, aligned_colored):
            full_normal += pn
            full_colored += pc
        print(f"  Total raw fused points: {len(full_normal.points):,}")

        # 6. Voxel De-duplication & SOR Denoising
        print(f"\n[Step 5] Voxel de-duplication (grid = {self.merge_voxel_size}m)...")
        dedup_normal = full_normal.voxel_down_sample(self.merge_voxel_size)
        dedup_colored = full_colored.voxel_down_sample(self.merge_voxel_size)
        print(f"  De-duplicated normal points:  {len(dedup_normal.points):,}")
        print(f"  De-duplicated colored points: {len(dedup_colored.points):,}")

        if self.enable_sor and len(dedup_normal.points) > 1000:
            print("  Applying Statistical Outlier Removal (SOR) filter...")
            _, ind_n = dedup_normal.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            dedup_normal = dedup_normal.select_by_index(ind_n)
            _, ind_c = dedup_colored.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            dedup_colored = dedup_colored.select_by_index(ind_c)
            print(f"  Cleaned deduplicated normal points:  {len(dedup_normal.points):,}")
            print(f"  Cleaned deduplicated colored points: {len(dedup_colored.points):,}")

        # 7. Write Deliverables & Multi-Sequence BEV
        prefix = f"fused_{len(sequences)}_streams"
        normal_full_ply = out_path / f"{prefix}_normal_full.ply"
        normal_dedup_ply = out_path / f"{prefix}_normal_merged.ply"
        colored_full_ply = out_path / f"{prefix}_colored_full.ply"
        colored_dedup_ply = out_path / f"{prefix}_colored_merged.ply"
        trajectory_bev_png = out_path / f"{prefix}_trajectory_bev.png"
        trajectories_json = out_path / f"{prefix}_camera_trajectories.json"
        meta_json = out_path / f"{prefix}_transform.json"

        print("\n[Step 6] Saving deliverables and multi-stream trajectory BEV...")
        o3d.io.write_point_cloud(str(normal_full_ply), full_normal)
        o3d.io.write_point_cloud(str(normal_dedup_ply), dedup_normal)
        o3d.io.write_point_cloud(str(colored_full_ply), full_colored)
        o3d.io.write_point_cloud(str(colored_dedup_ply), dedup_colored)

        bev_plane = write_multi_sequence_bev(
            path=trajectory_bev_png,
            camera_trajectories=transformed_trajectories,
            palette=palette,
            point_counts={s: len(p.points) for s, p in aligned_pcds_map.items()},
            size=self.bev_size,
        )
        print(f"  Exported Multi-Stream BEV Trajectory: {trajectory_bev_png} on {bev_plane.upper()} plane")

        # Save camera trajectory coordinates JSON
        traj_payload = {
            "anchor": anchor,
            "bev_plane": bev_plane,
            "sequences": {
                sid: {
                    "poses_count": len(poses),
                    "color_rgb": palette[sid].tolist(),
                    "camera_centers_3d": poses[:, :3, 3].tolist(),
                }
                for sid, poses in transformed_trajectories.items()
            },
        }
        with open(trajectories_json, "w", encoding="utf-8") as f:
            json.dump(traj_payload, f, indent=2, ensure_ascii=False)

        total_time = round(time.time() - t_start, 2)

        meta_data = {
            "method": "Method 2: Multi-View Sim(3) Spanning Tree + PGO Fusion (LightGlue + Umeyama + VGICP)",
            "anchor_sequence": anchor,
            "merge_voxel_size_m": self.merge_voxel_size,
            "total_raw_points": len(full_normal.points),
            "dedup_points": len(dedup_normal.points),
            "pgo_optimization": pgo_info,
            "pairwise_overlap_metrics": overlap_metrics,
            "sequences": point_stats,
            "tree_edges": [
                {
                    "src": e.src,
                    "tgt": e.tgt,
                    "inliers": e.inliers,
                    "inlier_ratio": round(e.inlier_ratio, 4),
                    "scale": round(e.scale, 6),
                    "rmse_mm": round(e.rmse * 1000, 2),
                }
                for e in mst_edges
            ],
            "deliverables": {
                "normal_full": str(normal_full_ply),
                "normal_dedup": str(normal_dedup_ply),
                "colored_full": str(colored_full_ply),
                "colored_dedup": str(colored_dedup_ply),
                "trajectory_bev": str(trajectory_bev_png),
                "camera_trajectories": str(trajectories_json),
            },
            "total_wall_time_s": total_time,
        }

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved Normal Fused PLY (Dedup):   {normal_dedup_ply} ({normal_dedup_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Saved Colored Fused PLY (Dedup):  {colored_dedup_ply} ({colored_dedup_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Saved Multi-Stream BEV Trajectory:{trajectory_bev_png}")
        print(f"Saved Trajectories JSON:          {trajectories_json}")
        print(f"Saved Metadata JSON:              {meta_json}")
        print("=" * 80)
        print(f"Fusion Completed Successfully in {total_time:.2f}s!")
        print("=" * 80)
        return meta_data


def main() -> None:
    parser = argparse.ArgumentParser(description="General-purpose Method 2 multi-stream point cloud fusion")
    parser.add_argument(
        "--recon-dirs",
        nargs="+",
        required=True,
        help="List of reconstruction directories (containing reconstruction.ply, world_points.pt, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/alignment/general_fusion",
        help="Directory to save fused deliverables",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help="Optional anchor sequence ID; if omitted, automatically selected based on connectivity",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Voxel size for de-duplication in meters (default: 0.015m)",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=5,
        help="Sampling stride for keyframe matching",
    )
    parser.add_argument(
        "--pgo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Sim(3) Global Pose Graph Optimization over all loop closures",
    )
    parser.add_argument(
        "--sor-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Statistical Outlier Removal (SOR) on merged point cloud",
    )
    parser.add_argument(
        "--multi-scale-icp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-scale coarse-to-fine VGICP refinement",
    )
    parser.add_argument(
        "--bev-size",
        type=int,
        default=1600,
        help="Resolution of the multi-stream BEV trajectory canvas",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    engine = Method2MultiViewFusion(
        device=args.device,
        keyframe_stride=args.keyframe_stride,
        merge_voxel_size=args.voxel_size,
        enable_pgo=args.pgo,
        enable_sor=args.sor_filter,
        multi_scale_icp=args.multi_scale_icp,
        bev_size=args.bev_size,
    )
    engine.fuse(
        recon_dirs=args.recon_dirs,
        output_dir=args.output_dir,
        anchor_id=args.anchor,
    )


if __name__ == "__main__":
    main()
