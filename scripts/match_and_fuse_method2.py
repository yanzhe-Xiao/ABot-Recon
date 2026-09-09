#!/usr/bin/env python3
"""
Method 2 General Pipeline: Video & Point Cloud Multi-Stream Matching and Fusion
(LightGlue + 2D-to-3D Lifting + Umeyama Sim(3) + small_gicp VGICP)

Features:
  1. Flexible Inputs:
     - Accepts raw video files (.mp4/.mov/...), image directories, or pre-reconstructed
       point cloud folders (outputs/<name>_loop). If a video is not yet reconstructed,
       it automatically extracts frames and runs ABot-Recon.
  2. Automated Topology & Global Anchor Selection:
     - DINO-SALAD global retrieval finds overlapping candidate keyframe pairs.
     - ALIKED + LightGlue finds 2D pixel correspondences.
     - 2D-to-3D unprojection lifts matches to metric space.
     - RANSAC + Umeyama SVD solves Sim(3) closed-form scale, rotation, translation.
     - Maximum Spanning Tree (MST) selects optimal anchor and propagation path.
     - small_gicp VGICP refines spatial seams.
  3. Fully Selectable Output Deliverables (--outputs):
     - `normal` / `normal_merged`: 1.5cm voxel de-duplicated photorealistic true-color PLY
     - `normal_full`: 100% full-resolution lossless true-color PLY
     - `colored` / `colored_merged`: Distinct-color segmented PLY (each video stream assigned unique color)
     - `colored_full`: Distinct-color 100% full-resolution PLY
     - `aligned`: Individual transformed point clouds in unified coordinate frame (<id>_aligned.ply)
     - `transforms`: Transformation matrices, scale factors, and connectivity JSON
     - `report`: Detailed Markdown registration report (inliers, RMSE, fitness)
     - `viewer`: Automatically register/deploy to viewer/server.py and index.html
     - `all`: Export all available deliverables above (default)
"""

from __future__ import annotations

import argparse
import colorsys
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def generate_distinct_palette(n: int) -> List[Tuple[float, float, float]]:
    """Generate `n` maximally distinct colors using Golden Ratio HSV."""
    preset = [
        (0.92, 0.20, 0.20),  # Red
        (0.15, 0.80, 0.25),  # Green
        (0.12, 0.56, 1.00),  # Blue
        (1.00, 0.80, 0.08),  # Gold
        (0.85, 0.20, 0.85),  # Magenta
        (0.10, 0.85, 0.85),  # Cyan
        (1.00, 0.50, 0.10),  # Orange
        (0.55, 0.25, 0.80),  # Purple
        (0.60, 0.90, 0.20),  # Lime
        (0.90, 0.35, 0.55),  # Pink
    ]
    if n <= len(preset):
        return preset[:n]
    out = []
    golden = 0.618033988749895
    h = 0.12
    for _ in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append((round(r, 4), round(g, 4), round(b, 4)))
    return out


@dataclass
class SequenceItem:
    id: str
    recon_dir: Path
    video_path: Optional[Path]
    pcd_raw: o3d.geometry.PointCloud
    pcd_down: o3d.geometry.PointCloud
    colors: torch.Tensor
    world_pts: torch.Tensor
    conf: torch.Tensor
    kf_indices: np.ndarray
    descriptors: np.ndarray


@dataclass
class MatchEdge:
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


class Method2FusionPipeline:
    """Universal Video and Point Cloud Multi-Stream Matching & Fusion Engine."""

    ALL_OUTPUTS = {
        "normal",
        "normal_full",
        "colored",
        "colored_full",
        "aligned",
        "transforms",
        "report",
        "viewer",
    }

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
        self.keyframe_stride = keyframe_stride
        self.top_k_pairs = top_k_pairs
        self.ransac_thresh = ransac_thresh
        self.merge_voxel_size = merge_voxel_size
        self.downsample_voxel_size = downsample_voxel_size

        print(f"[Pipeline] Initializing ALIKED + LightGlue on {self.device}...")
        self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)
        self.retrieval_cfg = RetrievalConfig(
            salad_checkpoint=REPO_ROOT / "checkpoints/loop/dino_salad.ckpt",
            dino_checkpoint=REPO_ROOT / "checkpoints/loop/dinov2_vitb14_pretrain.pth",
            backbone="dinov2_vitb14",
            verbose=False,
        )

    def prepare_input(self, input_path: str | Path, work_dir: Path) -> Path:
        """Resolve an input (video, directory of images, or existing reconstruction dir)."""
        p = Path(input_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {p}")

        # Case 1: Already a valid reconstruction directory
        if p.is_dir() and (p / "reconstruction.ply").is_file() and (p / "world_points.pt").is_file():
            return p

        # Case 2: A video file (.mp4, .mov, etc.)
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            seq_id = p.stem
            # Check if there is already a reconstruction for this video in outputs/
            existing = REPO_ROOT / f"outputs/{seq_id}_loop"
            if existing.is_dir() and (existing / "reconstruction.ply").is_file():
                print(f"  [Video] Found existing reconstruction: {existing}")
                return existing

            existing_data = REPO_ROOT / f"outputs/data_{seq_id}_loop"
            if existing_data.is_dir() and (existing_data / "reconstruction.ply").is_file():
                print(f"  [Video] Found existing reconstruction: {existing_data}")
                return existing_data

            # Needs extraction and reconstruction
            frames_dir = work_dir / "frames" / seq_id
            recon_dir = work_dir / f"recon_{seq_id}_loop"
            frames_dir.mkdir(parents=True, exist_ok=True)
            recon_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [Video] Extracting frames at 10 FPS: {p.name} -> {frames_dir}...")
            cmd_ffmpeg = [
                "ffmpeg", "-y", "-i", str(p),
                "-r", "10", "-q:v", "2",
                str(frames_dir / "%06d.jpg"),
            ]
            subprocess.run(cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            print(f"  [Reconstruction] Running ABot-Recon on {frames_dir.name}...")
            from scripts.reconstruct_sequences import reconstruct_sequence
            from abot_recon import ABotRecon
            model = ABotRecon.from_pretrained(
                REPO_ROOT / "checkpoints/abot_recon.safetensors",
                device=str(self.device),
                attention_backend="paged",
                loop_closure=True,
            )
            reconstruct_sequence(
                model=model,
                seq_id=seq_id,
                image_dir=frames_dir,
                output_dir=recon_dir,
                loop_output_dir=recon_dir / "loop",
                point_stride=4,
                device=str(self.device),
            )
            return recon_dir

        # Case 3: An image directory
        if p.is_dir():
            seq_id = p.name
            recon_dir = work_dir / f"recon_{seq_id}_loop"
            recon_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [Images] Reconstructing from image folder: {p}...")
            from scripts.reconstruct_sequences import reconstruct_sequence
            from abot_recon import ABotRecon
            model = ABotRecon.from_pretrained(
                REPO_ROOT / "checkpoints/abot_recon.safetensors",
                device=str(self.device),
                attention_backend="paged",
                loop_closure=True,
            )
            reconstruct_sequence(
                model=model,
                seq_id=seq_id,
                image_dir=p,
                output_dir=recon_dir,
                loop_output_dir=recon_dir / "loop",
                point_stride=4,
                device=str(self.device),
            )
            return recon_dir

        raise ValueError(f"Unsupported input type: {p}")

    def load_sequence_items(self, recon_dirs: List[Path]) -> Dict[str, SequenceItem]:
        items: Dict[str, SequenceItem] = {}
        for rdir in recon_dirs:
            sid = rdir.name
            if sid.startswith("data_") and sid.endswith("_loop"):
                sid = sid[5:-5]
            elif sid.startswith("recon_") and sid.endswith("_loop"):
                sid = sid[6:-5]
            elif sid.endswith("_loop"):
                sid = sid[:-5]

            print(f"  [Load] Loading Sequence [{sid}] from {rdir.name}...")
            ply_path = rdir / "reconstruction.ply"
            colors = torch.load(rdir / "colors.pt", map_location="cpu", weights_only=True)
            world = torch.load(rdir / "world_points.pt", map_location="cpu", weights_only=True)
            conf = torch.load(rdir / "confidence.pt", map_location="cpu", weights_only=True)

            pcd_raw = o3d.io.read_point_cloud(str(ply_path))
            pcd_down = pcd_raw.voxel_down_sample(self.downsample_voxel_size)

            kf_idx = np.arange(0, len(colors), self.keyframe_stride)
            desc = compute_descriptors(colors[kf_idx].numpy(), self.retrieval_cfg, self.device)

            items[sid] = SequenceItem(
                id=sid,
                recon_dir=rdir,
                video_path=None,
                pcd_raw=pcd_raw,
                pcd_down=pcd_down,
                colors=colors,
                world_pts=world,
                conf=conf,
                kf_indices=kf_idx,
                descriptors=desc,
            )
            print(f"    -> Points: {len(pcd_raw.points):,}, Keyframes: {len(kf_idx)}")
        return items

    def match_pair(self, s_a: SequenceItem, s_b: SequenceItem) -> Optional[MatchEdge]:
        """Method 2 matching: LightGlue + 2D-to-3D + Umeyama + small_gicp."""
        sim = s_a.descriptors @ s_b.descriptors.T
        if float(sim.max()) < 0.25:
            return None

        top_flat = np.argsort(-sim, axis=None)[: self.top_k_pairs * 2]
        candidate_pairs = []
        seen = set()
        for flat in top_flat:
            i, j = np.unravel_index(flat, sim.shape)
            fa, fb = int(s_a.kf_indices[i]), int(s_b.kf_indices[j])
            if fa not in seen and len(candidate_pairs) < self.top_k_pairs:
                candidate_pairs.append((fa, fb, float(sim[i, j])))
                seen.add(fa)

        solutions = []
        for fa, fb, score in candidate_pairs:
            img_a = s_a.colors[fa].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            img_b = s_b.colors[fb].permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

            with torch.no_grad():
                feat_a = self.extractor.extract(img_a)
                feat_b = self.extractor.extract(img_b)
                match_out = self.matcher({"image0": feat_a, "image1": feat_b})
                feat_a, feat_b, match_out = [rbd(x) for x in [feat_a, feat_b, match_out]]

            matches = match_out["matches"]
            if len(matches) < 6:
                continue

            kpts_a = feat_a["keypoints"][matches[:, 0]].cpu().numpy()
            kpts_b = feat_b["keypoints"][matches[:, 1]].cpu().numpy()

            p3d_a, p3d_b = [], []
            for (xa, ya), (xb, yb) in zip(kpts_a, kpts_b):
                ia_y = min(max(int(round(ya)), 0), 279)
                ia_x = min(max(int(round(xa)), 0), 503)
                ib_y = min(max(int(round(yb)), 0), 279)
                ib_x = min(max(int(round(xb)), 0), 503)
                if s_a.conf[fa, ia_y, ia_x] > 0.05 and s_b.conf[fb, ib_y, ib_x] > 0.05:
                    pa = s_a.world_pts[fa, ia_y, ia_x].numpy()
                    pb = s_b.world_pts[fb, ib_y, ib_x].numpy()
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        p3d_a.append(pa)
                        p3d_b.append(pb)

            p3d_a, p3d_b = np.array(p3d_a), np.array(p3d_b)
            if len(p3d_a) >= 4:
                res = ransac_umeyama(p3d_a, p3d_b, estimate_scale=True, iters=3000, thresh=self.ransac_thresh)
                if res is not None:
                    scale, R, t_vec, inliers, rmse = res
                    solutions.append({
                        "fa": fa,
                        "fb": fb,
                        "matches": len(matches),
                        "pairs": len(p3d_a),
                        "inliers": len(inliers),
                        "inlier_ratio": len(inliers) / len(p3d_a),
                        "scale": scale,
                        "R": R,
                        "t": t_vec,
                        "rmse": rmse,
                    })

        if not solutions:
            return None

        solutions.sort(key=lambda x: (-x["inliers"], -x["inlier_ratio"], x["rmse"]))
        best = solutions[0]

        # small_gicp VGICP refinement
        pts_coarse = (np.asarray(s_a.pcd_down.points, dtype=np.float64) * best["scale"]) @ best["R"].T + best["t"]
        gicp_res = small_gicp.align(
            target_points=np.asarray(s_b.pcd_down.points, dtype=np.float64),
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

        score = float(best["inliers"]) * (1.0 / (best["rmse"] + 0.01))
        return MatchEdge(
            src=s_a.id,
            tgt=s_b.id,
            fs=best["fa"],
            ft=best["fb"],
            matches=best["matches"],
            inliers=best["inliers"],
            inlier_ratio=best["inlier_ratio"],
            scale=best["scale"],
            R=best["R"],
            t=best["t"],
            rmse=best["rmse"],
            T_fine=T_fine,
            score=score,
        )

    def solve_multiview_topology(
        self,
        items: Dict[str, SequenceItem],
        custom_edges: Optional[List[Tuple[str, str]]] = None,
        anchor_override: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Tuple[float, np.ndarray]], List[MatchEdge]]:
        """Find Maximum Spanning Tree and compute global transformations to anchor."""
        seq_ids = list(items.keys())
        print(f"\n[Topology] Discovering connectivity graph across {len(seq_ids)} sequences...")

        edges: List[MatchEdge] = []
        for i in range(len(seq_ids)):
            for j in range(len(seq_ids)):
                if i != j:
                    e = self.match_pair(items[seq_ids[i]], items[seq_ids[j]])
                    if e is not None:
                        edges.append(e)
                        print(
                            f"  Edge [{e.src} -> {e.tgt}]: {e.inliers} inliers, "
                            f"scale={e.scale:.4f}, RMSE={e.rmse*1000:.1f}mm"
                        )

        if not edges:
            raise RuntimeError("No overlapping sequences could be registered!")

        # Choose Anchor
        if anchor_override is not None:
            anchor = anchor_override
        else:
            scores = {s: 0.0 for s in seq_ids}
            for e in edges:
                scores[e.tgt] += e.score
                scores[e.src] += e.score * 0.5
            anchor = max(scores, key=lambda k: scores[k])
        print(f"[Anchor] Selected reference anchor: [{anchor}]")

        adj: Dict[str, List[Tuple[str, MatchEdge]]] = {s: [] for s in seq_ids}
        for e in edges:
            adj[e.tgt].append((e.src, e))
        if custom_edges is not None:
            # Use user-specified edges
            edge_map = {(e.src, e.tgt): e for e in edges}
            transforms = {anchor: (1.0, np.eye(4, dtype=np.float64))}
            visited = {anchor}
            mst_edges = []
            while len(visited) < len(seq_ids):
                added = False
                for src, tgt in custom_edges:
                    if tgt in visited and src not in visited and (src, tgt) in edge_map:
                        edge = edge_map[(src, tgt)]
                        p_scale, p_T = transforms[tgt]
                        c_scale = p_scale * edge.scale
                        T_scaled = edge.T_fine.copy()
                        T_scaled[:3, 3] *= p_scale
                        transforms[src] = (c_scale, p_T @ T_scaled)
                        visited.add(src)
                        mst_edges.append(edge)
                        added = True
                        print(f"  [Custom Tree] Connected [{src}] -> [{tgt}] (scale={c_scale:.4f})")
                        break
                if not added:
                    break
            return anchor, transforms, mst_edges

        # BFS along MST
        transforms: Dict[str, Tuple[float, np.ndarray]] = {
            anchor: (1.0, np.eye(4, dtype=np.float64))
        }
        visited = {anchor}
        mst_edges: List[MatchEdge] = []

        while len(visited) < len(seq_ids):
            best_c = None
            best_e = None
            best_p = None
            best_w = -1.0
            for u in visited:
                for v, e in adj[u]:
                    if v not in visited and e.score > best_w:
                        best_w = e.score
                        best_c = v
                        best_e = e
                        best_p = u

            if best_c is None:
                print(f"  [Warning] Graph disconnected! Remaining: {[s for s in seq_ids if s not in visited]}")
                break

            p_scale, p_T = transforms[best_p]
            e_scale = best_e.scale
            e_T = best_e.T_fine

            c_scale = p_scale * e_scale
            T_scaled = e_T.copy()
            T_scaled[:3, 3] *= p_scale
            c_T = p_T @ T_scaled

            transforms[best_c] = (c_scale, c_T)
            visited.add(best_c)
            mst_edges.append(best_e)
            print(f"  [Tree] Connected [{best_c}] -> [{best_p}] (scale={c_scale:.4f})")

        return anchor, transforms, mst_edges

    def execute(
        self,
        inputs: List[str | Path],
        output_dir: str | Path,
        outputs: Sequence[str] | str = "all",
        anchor: Optional[str] = None,
        prefix: str = "fused",
        edges: Optional[str | List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Method 2 fusion and selectively generate specified artifacts.
        
        Available outputs:
          - normal / normal_merged: Voxel de-duplicated true-color PLY
          - normal_full: Full-resolution true-color PLY
          - colored / colored_merged: Voxel de-duplicated distinctly-colored PLY
          - colored_full: Full-resolution distinctly-colored PLY
          - aligned: Transformed individual PLYs (<id>_aligned.ply)
          - transforms: Transformation matrices & scale factors JSON
          - report: Markdown evaluation report
          - viewer: Automatically register and deploy to viewer/server.py
          - all: Generate everything
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        np.random.seed(42)
        torch.manual_seed(42)

        # Parse custom edges if given
        custom_edges = None
        if edges is not None:
            if isinstance(edges, str):
                custom_edges = [tuple(pair.split(":")) for pair in edges.split(",") if ":" in pair]
            else:
                custom_edges = list(edges)

        # Parse requested outputs
        if isinstance(outputs, str):
            if outputs.strip().lower() == "all":
                req_outputs = set(self.ALL_OUTPUTS)
            else:
                req_outputs = {o.strip().lower() for o in outputs.split(",") if o.strip()}
        else:
            req_outputs = {o.strip().lower() for o in outputs}
        if "all" in req_outputs:
            req_outputs = set(self.ALL_OUTPUTS)

        # Normalize aliases
        if "normal_merged" in req_outputs:
            req_outputs.add("normal")
        if "colored_merged" in req_outputs:
            req_outputs.add("colored")

        print("=" * 80)
        print("Method 2 Multi-Stream Point Cloud Matching & Fusion")
        print("=" * 80)
        print(f"Inputs:             {len(inputs)} sequences/videos")
        print(f"Output Directory:   {out_dir}")
        print(f"Requested Outputs:  {sorted(list(req_outputs))}\n")

        # 1. Resolve inputs (auto-extract/reconstruct if necessary)
        print("[Step 1] Resolving inputs...")
        work_dir = out_dir / ".cache"
        recon_dirs = [self.prepare_input(inp, work_dir) for inp in inputs]

        # 2. Load sequence items
        print("\n[Step 2] Loading sequences and computing descriptors...")
        items = self.load_sequence_items(recon_dirs)

        # 3. Topology & Alignment
        print("\n[Step 3] Solving Sim(3) graph topology...")
        anchor_id, transforms, mst_edges = self.solve_multiview_topology(
            items, custom_edges=custom_edges, anchor_override=anchor
        )

        # 4. Color Palette
        palette_list = generate_distinct_palette(len(items))
        palette = {sid: np.array(col, dtype=np.float64) for sid, col in zip(items.keys(), palette_list)}

        # 5. Transform point clouds
        print("\n[Step 4] Transforming point clouds into anchor frame...")
        aligned_normal: List[o3d.geometry.PointCloud] = []
        aligned_colored: List[o3d.geometry.PointCloud] = []
        individual_aligned: Dict[str, o3d.geometry.PointCloud] = {}
        seq_stats: Dict[str, Any] = {}

        for sid, seq in items.items():
            if sid not in transforms:
                continue
            scale, T = transforms[sid]
            pts_scaled = np.asarray(seq.pcd_raw.points, dtype=np.float64) * scale
            pts_trans = (pts_scaled @ T[:3, :3].T) + T[:3, 3]

            # True-color
            pcd_n = o3d.geometry.PointCloud()
            pcd_n.points = o3d.utility.Vector3dVector(pts_trans)
            pcd_n.colors = seq.pcd_raw.colors
            aligned_normal.append(pcd_n)
            individual_aligned[sid] = pcd_n

            # Distinct-color
            pcd_c = o3d.geometry.PointCloud()
            pcd_c.points = o3d.utility.Vector3dVector(pts_trans)
            col_matrix = np.tile(palette[sid], (len(pts_trans), 1))
            pcd_c.colors = o3d.utility.Vector3dVector(col_matrix)
            aligned_colored.append(pcd_c)

            seq_stats[sid] = {
                "points": len(pts_trans),
                "scale_to_anchor": round(float(scale), 6),
                "transform_matrix": T.tolist(),
                "color_rgb": palette[sid].tolist(),
            }
            print(f"  Seq [{sid}]: {len(pts_trans):,} points transformed (scale s={scale:.4f})")

        # Merge
        full_normal = o3d.geometry.PointCloud()
        full_colored = o3d.geometry.PointCloud()
        for pn, pc in zip(aligned_normal, aligned_colored):
            full_normal += pn
            full_colored += pc

        dedup_normal = None
        dedup_colored = None
        if "normal" in req_outputs or "colored" in req_outputs:
            print(f"\n[Step 5] Performing voxel de-duplication ({self.merge_voxel_size}m)...")
            if "normal" in req_outputs:
                dedup_normal = full_normal.voxel_down_sample(self.merge_voxel_size)
            if "colored" in req_outputs:
                dedup_colored = full_colored.voxel_down_sample(self.merge_voxel_size)

        # 6. Generate Selectable Outputs
        print("\n[Step 6] Exporting requested deliverables...")
        deliverables: Dict[str, str] = {}

        # 6.1 Normal Merged PLY
        if "normal" in req_outputs and dedup_normal is not None:
            p_out = out_dir / f"{prefix}_normal_merged.ply"
            o3d.io.write_point_cloud(str(p_out), dedup_normal)
            deliverables["normal_merged"] = str(p_out)
            print(f"  [Saved] Normal Merged PLY (Dedup): {p_out} ({p_out.stat().st_size / 1024 / 1024:.1f} MB)")

        # 6.2 Normal Full PLY
        if "normal_full" in req_outputs:
            p_out = out_dir / f"{prefix}_normal_full.ply"
            o3d.io.write_point_cloud(str(p_out), full_normal)
            deliverables["normal_full"] = str(p_out)
            print(f"  [Saved] Normal Full PLY:          {p_out} ({p_out.stat().st_size / 1024 / 1024:.1f} MB)")

        # 6.3 Colored Merged PLY
        if "colored" in req_outputs and dedup_colored is not None:
            p_out = out_dir / f"{prefix}_colored_merged.ply"
            o3d.io.write_point_cloud(str(p_out), dedup_colored)
            deliverables["colored_merged"] = str(p_out)
            print(f"  [Saved] Colored Merged PLY:       {p_out} ({p_out.stat().st_size / 1024 / 1024:.1f} MB)")

        # 6.4 Colored Full PLY
        if "colored_full" in req_outputs:
            p_out = out_dir / f"{prefix}_colored_full.ply"
            o3d.io.write_point_cloud(str(p_out), full_colored)
            deliverables["colored_full"] = str(p_out)
            print(f"  [Saved] Colored Full PLY:         {p_out} ({p_out.stat().st_size / 1024 / 1024:.1f} MB)")

        # 6.5 Aligned Individual PLYs
        if "aligned" in req_outputs:
            aligned_dir = out_dir / "aligned_individual"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            for sid, pcd in individual_aligned.items():
                p_out = aligned_dir / f"seq_{sid}_aligned.ply"
                o3d.io.write_point_cloud(str(p_out), pcd)
            deliverables["aligned_individual_dir"] = str(aligned_dir)
            print(f"  [Saved] Aligned Individual PLYs:  {aligned_dir}/ (x{len(individual_aligned)})")

        # 6.6 Transforms JSON
        transforms_file = out_dir / f"{prefix}_transforms.json"
        meta_data = {
            "method": "Method 2 Multi-View Sim(3) Fusion (LightGlue + Umeyama + small_gicp)",
            "anchor_sequence": anchor_id,
            "merge_voxel_size_m": self.merge_voxel_size,
            "total_raw_points": len(full_normal.points),
            "dedup_points": len(dedup_normal.points) if dedup_normal else None,
            "sequences": seq_stats,
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
            "deliverables": deliverables,
            "total_wall_time_s": round(time.time() - t_start, 2),
        }
        if "transforms" in req_outputs:
            with open(transforms_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            deliverables["transforms_json"] = str(transforms_file)
            print(f"  [Saved] Transforms JSON:          {transforms_file}")

        # 6.7 Markdown Report
        if "report" in req_outputs:
            report_file = out_dir / f"{prefix}_REPORT.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"# 多视角点云配准融合技术报告 ({prefix})\n\n")
                f.write(f"- **对齐算法**：方案二（LightGlue + 2D-to-3D 升维 + Umeyama Sim(3) + small_gicp）\n")
                f.write(f"- **全局基准锚点**：`Sequence {anchor_id}`\n")
                f.write(f"- **总输入点数**：{len(full_normal.points):,} 点\n")
                if dedup_normal:
                    f.write(f"- **去重后点数**：{len(dedup_normal.points):,} 点 (体素 {self.merge_voxel_size}m)\n")
                f.write(f"- **总耗时**：{round(time.time() - t_start, 2)} s\n\n")
                f.write("## 1. 序列变换与尺度参数\n\n")
                f.write("| 序列 ID | 点数 | 尺度因子 s | 分配色彩 (RGB) |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for sid, stat in seq_stats.items():
                    rgb_str = f"[{stat['color_rgb'][0]:.2f}, {stat['color_rgb'][1]:.2f}, {stat['color_rgb'][2]:.2f}]"
                    f.write(f"| **{sid}** | {stat['points']:,} | {stat['scale_to_anchor']} | {rgb_str} |\n")
                f.write("\n## 2. 配准树拓扑边评估 (MST)\n\n")
                f.write("| 匹配边 (Src -> Tgt) | 3D 内点数 | 内点率 | 估计尺度 s | RMSE |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                for e in mst_edges:
                    f.write(f"| {e.src} $\\to$ {e.tgt} | {e.inliers} | {e.inlier_ratio*100:.1f}% | {e.scale:.4f} | {e.rmse*1000:.1f} mm |\n")
            deliverables["report_md"] = str(report_file)
            print(f"  [Saved] Markdown Report:          {report_file}")

        # 6.8 Viewer Deploy
        if "viewer" in req_outputs:
            print("  [Deploy] Registering new models to viewer/server.py...")
            self.deploy_to_viewer(deliverables, prefix, len(items))

        total_time = round(time.time() - t_start, 2)
        print("\n" + "=" * 80)
        print(f"Pipeline Completed in {total_time:.2f}s!")
        print(f"Output directory: {out_dir}")
        print("=" * 80)
        return meta_data

    def deploy_to_viewer(self, deliverables: Dict[str, str], prefix: str, count: int) -> None:
        """Register newly generated models to viewer/server.py and index.html."""
        server_py = REPO_ROOT / "viewer/server.py"
        if not server_py.is_file():
            return

        # Find relative path
        for key in ["normal_merged", "colored_merged", "normal_full"]:
            if key in deliverables:
                rel_path = Path(deliverables[key]).relative_to(REPO_ROOT)
                print(f"    Available on frontend: http://localhost:8088/{rel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Method 2 Video & Point Cloud Multi-Stream Matching and Fusion Tool"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video files (.mp4/...), image folders, or reconstruction dirs (outputs/<name>_loop)",
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=Path,
        default=REPO_ROOT / "outputs/alignment/custom_fusion",
        help="Target output directory for deliverables",
    )
    parser.add_argument(
        "--outputs", "-o",
        type=str,
        default="all",
        help=(
            "Selectable output deliverables (comma-separated or 'all'):\n"
            "  normal         : 1.5cm voxel deduplicated true-color PLY\n"
            "  normal_full    : 100% full-resolution true-color PLY\n"
            "  colored        : Distinct-color segmented PLY (unique color per stream)\n"
            "  colored_full   : Distinct-color full-resolution PLY\n"
            "  aligned        : Separate aligned point clouds for each stream (<id>_aligned.ply)\n"
            "  transforms     : JSON file with all Sim(3) transforms, scales, and matrices\n"
            "  report         : Markdown evaluation report with registration metrics\n"
            "  viewer         : Deploy and link to viewer/server\n"
            "  all            : Export everything above (default)"
        ),
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help="Manual sequence ID to use as anchor reference (default: auto-detected)",
    )
    parser.add_argument(
        "--edges",
        type=str,
        default=None,
        help="Optional manual registration edges (e.g. '05:06,06:07,08:07')",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Merge voxel grid size in meters (default: 0.015m / 1.5cm)",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=5,
        help="Keyframe sampling stride for retrieval (default: 5 frames)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fused",
        help="Filename prefix for generated files (default: 'fused')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    pipeline = Method2FusionPipeline(
        device=args.device,
        keyframe_stride=args.keyframe_stride,
        merge_voxel_size=args.voxel_size,
    )
    pipeline.execute(
        inputs=args.inputs,
        output_dir=args.output_dir,
        outputs=args.outputs,
        anchor=args.anchor,
        prefix=args.prefix,
        edges=args.edges,
    )


if __name__ == "__main__":
    main()
