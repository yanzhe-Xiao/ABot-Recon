#!/usr/bin/env python3
"""
General-Purpose Multi-Sequence Point Cloud Fusion using Method 2
(LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp VGICP)

Features:
  1. Arbitrary N sequences: accepts any number of sequence directories.
  2. Automatic Graph Topology & Anchor Selection:
     - Global image retrieval (DINO-SALAD) to find cross-sequence candidate overlaps.
     - 2D matching (ALIKED + LightGlue) & 3D lifting to solve Sim(3) for valid edges.
     - Maximum Spanning Tree (MST) and auto-selection of the optimal anchor reference.
  3. Seamless Chained Transformation & VGICP Refinement:
     - Automatically propagates scales, rotations, translations along the spanning tree.
     - Multi-threaded small_gicp eliminates fine seam errors.
  4. Automatic Dual-Version Export:
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


@dataclass
class SequenceData:
    id: str
    dir_path: Path
    pcd_raw: o3d.geometry.PointCloud
    pcd_down: o3d.geometry.PointCloud
    colors_tensor: torch.Tensor
    world_tensor: torch.Tensor
    conf_tensor: torch.Tensor
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
    ):
        self.device = torch.device(device)
        self.keyframe_stride = max(1, keyframe_stride)
        self.top_k_pairs = max(1, top_k_pairs)
        self.ransac_thresh = ransac_thresh
        self.merge_voxel_size = merge_voxel_size
        self.downsample_voxel_size = downsample_voxel_size

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
                keyframe_indices=kf_idx,
                descriptors=desc,
            )
            print(f"    -> {len(pcd.points):,} raw points, {len(kf_idx)} keyframes indexed.")
        return sequences

    def match_pair(self, seq_s: SequenceData, seq_t: SequenceData) -> Optional[EdgeResult]:
        """Match 2D features between two sequences and solve Umeyama Sim(3) + small_gicp."""
        sim = seq_s.descriptors @ seq_t.descriptors.T
        max_sim = float(sim.max())
        if max_sim < 0.25:  # Skip completely disjoint pairs early
            return None

        top_flat = np.argsort(-sim, axis=None)[: self.top_k_pairs * 2]
        candidate_pairs = []
        seen = set()
        for flat in top_flat:
            i, j = np.unravel_index(flat, sim.shape)
            fs, ft = int(seq_s.keyframe_indices[i]), int(seq_t.keyframe_indices[j])
            if fs not in seen and len(candidate_pairs) < self.top_k_pairs:
                candidate_pairs.append((fs, ft, float(sim[i, j])))
                seen.add(fs)

        pair_solutions = []
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
            for (xa, ya), (xb, yb) in zip(kpts_s, kpts_t):
                ia_y, ia_x = min(max(int(round(ya)), 0), 279), min(max(int(round(xa)), 0), 503)
                ib_y, ib_x = min(max(int(round(yb)), 0), 279), min(max(int(round(xb)), 0), 503)
                if seq_s.conf_tensor[fs, ia_y, ia_x] > 0.05 and seq_t.conf_tensor[ft, ib_y, ib_x] > 0.05:
                    pa = seq_s.world_tensor[fs, ia_y, ia_x].numpy()
                    pb = seq_t.world_tensor[ft, ib_y, ib_x].numpy()
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        p_s.append(pa)
                        p_t.append(pb)

            p_s, p_t = np.array(p_s), np.array(p_t)
            if len(p_s) >= 4:
                res = ransac_umeyama(p_s, p_t, estimate_scale=True, iters=3000, thresh=self.ransac_thresh)
                if res is not None:
                    s, R, t_vec, inliers, rmse = res
                    inlier_ratio = len(inliers) / len(p_s)
                    pair_solutions.append({
                        "fs": fs,
                        "ft": ft,
                        "matches": len(matches),
                        "pairs": len(p_s),
                        "inliers": len(inliers),
                        "inlier_ratio": inlier_ratio,
                        "scale": s,
                        "R": R,
                        "t": t_vec,
                        "rmse": rmse,
                    })

        if not pair_solutions:
            return None

        # Sort by total inliers and inlier ratio
        pair_solutions.sort(key=lambda x: (-x["inliers"], -x["inlier_ratio"], x["rmse"]))
        best = pair_solutions[0]

        # Refine with small_gicp VGICP
        pts_coarse = (np.asarray(seq_s.pcd_down.points, dtype=np.float64) * best["scale"]) @ best["R"].T + best["t"]
        gicp_res = small_gicp.align(
            target_points=np.asarray(seq_t.pcd_down.points, dtype=np.float64),
            source_points=pts_coarse,
            init_T_target_source=np.eye(4, dtype=np.float64),
            registration_type="VGICP",
            voxel_resolution=0.08,
            downsampling_resolution=self.downsample_voxel_size,
            max_correspondence_distance=0.08,
            max_iterations=40,
            num_threads=8,
        )

        T_rigid = np.eye(4, dtype=np.float64)
        T_rigid[:3, :3] = best["R"]
        T_rigid[:3, 3] = best["t"]
        T_fine = gicp_res.T_target_source @ T_rigid

        # Edge score combines inliers and low RMSE
        edge_score = float(best["inliers"]) * (1.0 / (best["rmse"] + 0.01))

        return EdgeResult(
            src=seq_s.id,
            tgt=seq_t.id,
            fs=best["fs"],
            ft=best["ft"],
            matches=best["matches"],
            inliers=best["inliers"],
            inlier_ratio=best["inlier_ratio"],
            scale=best["scale"],
            R=best["R"],
            t=best["t"],
            rmse=best["rmse"],
            T_fine=T_fine,
            score=edge_score,
        )

    def build_spanning_tree(
        self,
        sequences: Dict[str, SequenceData],
        anchor_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Tuple[float, np.ndarray]], List[EdgeResult]]:
        """
        Build registration graph, choose optimal anchor sequence,
        and compute Maximum Spanning Tree (MST) transforms to anchor.
        """
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
        # Map: tgt -> list of (src, EdgeResult)
        adj: Dict[str, List[Tuple[str, EdgeResult]]] = {sid: [] for sid in seq_ids}
        for e in edges:
            adj[e.tgt].append((e.src, e))

        # BFS / Dijkstra from root to find highest-confidence transform for every node
        # Transforms map: node_id -> (scale_to_anchor, T_to_anchor)
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

            # Compute chained transformation: v -> parent -> root
            # p_parent = T_edge[:3,:3] * (s_edge * p_v) + T_edge[:3,3]
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

        return root, transforms_to_anchor, mst_edges

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
        print(f"Sequences Count:     {len(recon_dirs)}\n")

        # 1. Load data
        print("[Step 1] Loading sequences & computing descriptors...")
        sequences = self.load_sequences(recon_dirs)

        # 2. Build Spanning Tree & solve all transforms to anchor
        print("\n[Step 2] Building topology graph & solving Sim(3) transforms...")
        anchor, transforms, mst_edges = self.build_spanning_tree(sequences, anchor_id=anchor_id)

        # 3. Generate colors palette
        colors = generate_distinct_colors(len(sequences))
        palette = {sid: np.array(col, dtype=np.float64) for sid, col in zip(sequences.keys(), colors)}

        # 4. Transform raw point clouds
        print("\n[Step 3] Applying transformations to point clouds...")
        aligned_normal = []
        aligned_colored = []
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

            # Colored
            pcd_c = o3d.geometry.PointCloud()
            pcd_c.points = o3d.utility.Vector3dVector(pts_trans)
            col_matrix = np.tile(palette[sid], (len(pts_trans), 1))
            pcd_c.colors = o3d.utility.Vector3dVector(col_matrix)
            aligned_colored.append(pcd_c)

            point_stats[sid] = {
                "raw_points": len(pts_trans),
                "scale_to_anchor": round(float(scale), 6),
                "color_rgb": palette[sid].tolist(),
                "transform_matrix": T.tolist(),
            }
            print(f"  Seq [{sid}]: {len(pts_trans):,} points transformed (scale s={scale:.4f})")

        # 5. Merge point clouds
        print("\n[Step 4] Merging raw point clouds...")
        full_normal = o3d.geometry.PointCloud()
        full_colored = o3d.geometry.PointCloud()
        for pn, pc in zip(aligned_normal, aligned_colored):
            full_normal += pn
            full_colored += pc
        print(f"  Total raw fused points: {len(full_normal.points):,}")

        # 6. Voxel De-duplication
        print(f"\n[Step 5] Voxel de-duplication (grid = {self.merge_voxel_size}m)...")
        dedup_normal = full_normal.voxel_down_sample(self.merge_voxel_size)
        dedup_colored = full_colored.voxel_down_sample(self.merge_voxel_size)
        print(f"  De-duplicated normal points:  {len(dedup_normal.points):,}")
        print(f"  De-duplicated colored points: {len(dedup_colored.points):,}")

        # 7. Write PLY Deliverables
        prefix = f"fused_{len(sequences)}_streams"
        normal_full_ply = out_path / f"{prefix}_normal_full.ply"
        normal_dedup_ply = out_path / f"{prefix}_normal_merged.ply"
        colored_full_ply = out_path / f"{prefix}_colored_full.ply"
        colored_dedup_ply = out_path / f"{prefix}_colored_merged.ply"
        meta_json = out_path / f"{prefix}_transform.json"

        print("\n[Step 6] Saving deliverables to disk...")
        o3d.io.write_point_cloud(str(normal_full_ply), full_normal)
        o3d.io.write_point_cloud(str(normal_dedup_ply), dedup_normal)
        o3d.io.write_point_cloud(str(colored_full_ply), full_colored)
        o3d.io.write_point_cloud(str(colored_dedup_ply), dedup_colored)

        total_time = round(time.time() - t_start, 2)

        meta_data = {
            "method": "Method 2: Multi-View Sim(3) Spanning Tree Fusion (LightGlue + Umeyama + small_gicp)",
            "anchor_sequence": anchor,
            "merge_voxel_size_m": self.merge_voxel_size,
            "total_raw_points": len(full_normal.points),
            "dedup_points": len(dedup_normal.points),
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
            },
            "total_wall_time_s": total_time,
        }

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved Normal Fused PLY (Dedup):   {normal_dedup_ply} ({normal_dedup_ply.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Saved Colored Fused PLY (Dedup):  {colored_dedup_ply} ({colored_dedup_ply.stat().st_size / 1024 / 1024:.1f} MB)")
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
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    engine = Method2MultiViewFusion(
        device=args.device,
        keyframe_stride=args.keyframe_stride,
        merge_voxel_size=args.voxel_size,
    )
    engine.fuse(
        recon_dirs=args.recon_dirs,
        output_dir=args.output_dir,
        anchor_id=args.anchor,
    )


if __name__ == "__main__":
    main()
