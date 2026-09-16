#!/usr/bin/env python3
"""
3D Point Cloud Reconstruction with Dynamic Object Removal (Scheme 1: 2D Semantic Masking).
Integrates YOLO-seg with ABot-Recon monocular streaming 3D reconstruction with EXACT pixel alignment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image
import open3d as o3d
import subprocess
from scripts.export_reconstruction_ply import write_bev, write_binary_ply


def extract_frames_from_video(video_path: Path, frames_dir: Path, fps_target: float | None = None) -> list[Path]:
    """Extract frames from video into frames_dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frames_dir.glob("*.jpg"))
    if existing_frames:
        print(f"Found {len(existing_frames)} existing frames in {frames_dir}")
        return existing_frames

    print(f"Extracting frames from {video_path} to {frames_dir}...")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps_target is not None and fps_target > 0:
        cmd.extend(["-vf", f"fps={fps_target}"])
    cmd.extend(["-q:v", "2", str(frames_dir / "%06d.jpg")])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with error:\n{res.stderr}")

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    print(f"Extracted {len(frames)} frames successfully.")
    return frames

class AlignedDynamicMaskGenerator:
    """
    Generates dynamic object masks directly on preprocessed [H, W, 3] tensors,
    ensuring 100% pixel-perfect alignment with ABot-Recon pointmaps.
    """

    def __init__(
        self,
        model_name: str = "yolo11m-seg.pt",
        dynamic_classes: list[int] | None = None,
        conf_thresh: float = 0.15,
        dilate_kernel: int = 11,
        device: str = "cuda",
    ):
        self.model = YOLO(model_name)
        # Default: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 15: cat, 16: dog
        self.dynamic_classes = dynamic_classes if dynamic_classes is not None else [0, 1, 2, 3, 5, 7, 15, 16]
        self.conf_thresh = conf_thresh
        self.dilate_kernel = dilate_kernel
        self.device = device

    def predict_masks_on_preprocessed(
        self, preprocessed_rgbs: list[np.ndarray], batch_size: int = 32
    ) -> list[np.ndarray]:
        """
        preprocessed_rgbs: list of uint8 ndarray [H, W, 3] (RGB, exactly matching ABot-Recon input)
        Returns: list of boolean ndarray [H, W], True = Static, False = Dynamic (filtered)
        """
        if not preprocessed_rgbs:
            return []
        h, w = preprocessed_rgbs[0].shape[:2]
        raw_dyn_masks = []

        # Batched YOLO segmentation inference
        for i in range(0, len(preprocessed_rgbs), batch_size):
            batch = preprocessed_rgbs[i : i + batch_size]
            results = self.model.predict(
                batch,
                batch=len(batch),
                classes=self.dynamic_classes,
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
            for res in results:
                if res.masks is not None and len(res.masks.data) > 0:
                    m_gpu = (res.masks.data.sum(dim=0) > 0).float().unsqueeze(0).unsqueeze(0)
                    if m_gpu.shape[-2:] != (h, w):
                        m_gpu = torch.nn.functional.interpolate(m_gpu, size=(h, w), mode="nearest")
                    dyn_mask = m_gpu.squeeze().to(torch.uint8).cpu().numpy()
                else:
                    dyn_mask = np.zeros((h, w), dtype=np.uint8)
                raw_dyn_masks.append(dyn_mask)
        # Temporal smoothing: if frame i-1 and i+1 have dynamic mask, propagate to frame i
        smoothed_dyn_masks = []
        n_frames = len(raw_dyn_masks)
        for i in range(n_frames):
            cur_mask = raw_dyn_masks[i].copy()
            if np.sum(cur_mask) == 0 and 0 < i < n_frames - 1:
                prev_mask = raw_dyn_masks[i - 1]
                next_mask = raw_dyn_masks[i + 1]
                if np.sum(prev_mask) > 0 and np.sum(next_mask) > 0:
                    cur_mask = np.bitwise_or(prev_mask, next_mask)

            # Morphological dilation to thoroughly eliminate edge bleeding/hairs
            if self.dilate_kernel > 0 and np.any(cur_mask):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel)
                )
                cur_mask = cv2.dilate(cur_mask, kernel)

            smoothed_dyn_masks.append(cur_mask == 0)

        return smoothed_dyn_masks


def export_colored_pointcloud(
    world_points: torch.Tensor,
    colors: torch.Tensor,
    mask: torch.Tensor | None,
    output_ply: Path,
    point_stride: int = 2,
) -> int:
    """Subsamples and exports valid 3D points + RGB colors to a binary PLY file."""
    pts = world_points[:, ::point_stride, ::point_stride, :].reshape(-1, 3)
    cls = colors[:, ::point_stride, ::point_stride, :].reshape(-1, 3)

    valid = torch.isfinite(pts).all(dim=-1)
    if mask is not None:
        sub_mask = mask[:, ::point_stride, ::point_stride].reshape(-1)
        valid = valid & sub_mask

    pts_valid = pts[valid].cpu().numpy().astype(np.float32)
    cls_valid = cls[valid].cpu().numpy().astype(np.uint8)

    empty_edges = np.empty((0, 2), dtype=np.int32)
    empty_edge_colors = np.empty((0, 3), dtype=np.uint8)
    write_binary_ply(output_ply, pts_valid, cls_valid, empty_edges, empty_edge_colors)
    return len(pts_valid)


def main():
    parser = argparse.ArgumentParser(description="ABot-Recon with Dynamic Mask Filtering")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Input video file or image directory",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Input frames directory (legacy alias)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints",
        help="Checkpoint directory or path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/glasshouse_dynamic_removal_scheme1"),
        help="Output directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--loop-closure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-filter", "--dymaic-filter", action=argparse.BooleanOptionalAction, dest="dynamic_filter", default=True, help="Enable dynamic object removal")
    parser.add_argument("--yolo-model", default="yolo11m-seg.pt")
    parser.add_argument("--conf-thresh", type=float, default=0.15)
    parser.add_argument("--dilate-kernel", type=int, default=11)
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for frame extraction from video")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input or args.image_dir or Path("data/glasshouse_stride5/images")

    if input_path.is_file():
        frames_dir = args.output_dir / "frames"
        images = extract_frames_from_video(input_path, frames_dir, fps_target=args.fps)
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not images:
        raise ValueError(f"No images found in {input_path}")

    print(f"[1/4] Found {len(images)} images in {input_path}")

    # 1. Preprocess images exactly as ABot-Recon does
    print("[2/4] Preprocessing frames & generating aligned 2D dynamic masks with YOLO-seg...")
    preprocessed_tensors = []
    preprocessed_rgbs = []
    for img_path in images:
        with Image.open(img_path) as img:
            tensor, _ = preprocess_image(img, height=280, width=504)
        preprocessed_tensors.append(tensor)
        rgb_uint8 = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        preprocessed_rgbs.append(rgb_uint8)

    if args.dynamic_filter:
        mask_gen = AlignedDynamicMaskGenerator(
            model_name=args.yolo_model,
            conf_thresh=args.conf_thresh,
            dilate_kernel=args.dilate_kernel,
            device=args.device,
        )
        t0 = time.time()
        static_masks_np = mask_gen.predict_masks_on_preprocessed(preprocessed_rgbs)
        mask_time = time.time() - t0
        print(f"Aligned dynamic mask generation finished in {mask_time:.2f}s ({len(images)/mask_time:.1f} FPS)")

        # Save visual debug samples
        vis_dir = args.output_dir / "aligned_masks_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(images) // 20)
        for idx in range(0, len(images), step):
            rgb_vis = preprocessed_rgbs[idx].copy()
            dyn_pixels = ~static_masks_np[idx]
            rgb_vis[dyn_pixels] = [255, 0, 0]  # Mark dynamic as red
            out_vis = cv2.addWeighted(preprocessed_rgbs[idx], 0.5, rgb_vis, 0.5, 0)
            cv2.imwrite(str(vis_dir / f"aligned_vis_{images[idx].name}"), cv2.cvtColor(out_vis, cv2.COLOR_RGB2BGR))
        vis_dir_path = str(vis_dir)
    else:
        static_masks_np = [np.ones((280, 504), dtype=bool) for _ in range(len(images))]
        mask_time = 0.0
        vis_dir_path = "N/A (dynamic filter disabled)"
    # 2. Run ABot-Recon Reconstruction
    print("[3/4] Running ABot-Recon 3D reconstruction...")
    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        attention_backend="auto",
        loop_closure=args.loop_closure,
        loop_salad_checkpoint=Path("checkpoints/loop/dino_salad.ckpt"),
        loop_dino_checkpoint=Path("checkpoints/loop/dinov2_vitb14_pretrain.pth"),
        loop_output_dir=args.output_dir / "loop",
        local_files_only=True,
    )

    t1 = time.time()
    result = model.infer(
        images,
        output_local_points=True,
        output_world_points=True,
        output_confidence=True,
        confidence_threshold=args.confidence_threshold,
        loop_closure=args.loop_closure,
    )
    recon_time = time.time() - t1
    print(f"ABot-Recon finished in {recon_time:.2f}s ({len(images)/recon_time:.1f} FPS)")

    # 3. Save standard artifacts for multi-view registration and viewer compatibility
    np.save(args.output_dir / "camera_poses.npy", result.camera_poses.cpu().numpy())
    np.save(args.output_dir / "relative_poses.npy", result.relative_poses.cpu().numpy())
    torch.save(result.world_points.cpu(), args.output_dir / "world_points.pt")
    if result.local_points is not None:
        torch.save(result.local_points.cpu(), args.output_dir / "local_points.pt")
    colors = torch.stack([torch.from_numpy(rgb) for rgb in preprocessed_rgbs])  # [N, 280, 504, 3]
    torch.save(colors.cpu(), args.output_dir / "colors.pt")
    if result.confidence is not None:
        torch.save(result.confidence.cpu(), args.output_dir / "confidence.pt")
    static_masks_tensor = torch.stack([torch.from_numpy(m) for m in static_masks_np]).to(result.local_points.device)  # [N, 280, 504]
    torch.save(static_masks_tensor.cpu(), args.output_dir / "static_masks.pt")
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)

    world_pts = result.world_points

    # 4. Export Baseline, Filtered (Static Background), and Removed Dynamic Point Clouds
    print("[4/4] Exporting Baseline vs Filtered vs Dynamic-Only 3D Point Clouds...")
    baseline_ply = args.output_dir / "reconstruction_baseline_with_dynamic.ply"
    filtered_ply = args.output_dir / "reconstruction_scheme1_filtered.ply"
    dynamic_only_ply = args.output_dir / "reconstruction_removed_dynamic_only.ply"

    # Baseline mask: only finite & confidence
    baseline_valid_mask = torch.isfinite(world_pts).all(dim=-1)
    if result.confidence_mask is not None:
        baseline_valid_mask = baseline_valid_mask & result.confidence_mask

    baseline_points_count = export_colored_pointcloud(
        world_pts, colors, baseline_valid_mask, baseline_ply, point_stride=args.point_stride
    )

    # Filtered mask: baseline & static mask (strictly removes all dynamic entity points)
    filtered_valid_mask = baseline_valid_mask & static_masks_tensor
    filtered_points_count = export_colored_pointcloud(
        world_pts, colors, filtered_valid_mask, filtered_ply, point_stride=args.point_stride
    )

    # Dynamic only mask: baseline & (~static mask)
    dynamic_only_mask = baseline_valid_mask & (~static_masks_tensor)
    dynamic_only_points_count = export_colored_pointcloud(
        world_pts, colors, dynamic_only_mask, dynamic_only_ply, point_stride=args.point_stride
    )

    # Also save standard reconstruction.ply as filtered_ply
    shutil.copyfile(filtered_ply, args.output_dir / "reconstruction.ply")

    # BEV Trajectory map
    bev_png = args.output_dir / "trajectory_bev.png"
    poses_np = result.camera_poses.cpu().numpy()
    plane = write_bev(bev_png, poses_np, 1600, "auto")
    print(f"Exported BEV Trajectory: {bev_png} ({len(poses_np)} poses on {plane.upper()} plane)")

    # Filtered & Denoised PLY via Open3D
    clean_ply_path = args.output_dir / "reconstruction_clean.ply"
    if filtered_ply.exists() and filtered_ply.stat().st_size > 0:
        print("Post-processing point cloud with Open3D statistical outlier removal...")
        pcd = o3d.io.read_point_cloud(str(filtered_ply))
        if len(pcd.points) > 0:
            pcd_down = pcd.voxel_down_sample(voxel_size=0.015)
            cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
            clean_pcd = pcd_down.select_by_index(ind)
            o3d.io.write_point_cloud(str(clean_ply_path), clean_pcd, write_ascii=False)
            print(f"Exported Cleaned PLY: {clean_ply_path} ({len(clean_pcd.points):,} points, {clean_ply_path.stat().st_size / 1024 / 1024:.2f} MB)")

    dynamic_points_removed = baseline_points_count - filtered_points_count
    reduction_pct = (dynamic_points_removed / max(1, baseline_points_count)) * 100.0

    report = {
        "total_frames": len(images),
        "mask_generation_time_s": round(mask_time, 2),
        "recon_time_s": round(recon_time, 2),
        "baseline_points": baseline_points_count,
        "filtered_points": filtered_points_count,
        "dynamic_only_points": dynamic_only_points_count,
        "dynamic_points_removed": dynamic_points_removed,
        "dynamic_removal_ratio_pct": round(reduction_pct, 2),
        "baseline_ply": str(baseline_ply),
        "filtered_ply": str(filtered_ply),
        "dynamic_only_ply": str(dynamic_only_ply),
        "reconstruction_ply": str(args.output_dir / "reconstruction.ply"),
    }

    with (args.output_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("           Dynamic Object Removal Evaluation           ")
    print("=======================================================")
    print(f"Total Video Frames Processed: {len(images)}")
    print(f"1. Baseline Point Cloud (With Dynamic Objects): {baseline_points_count:,} points -> {baseline_ply.name}")
    print(f"2. Scheme 1 Point Cloud (Clean Static Map):     {filtered_points_count:,} points -> {filtered_ply.name}")
    print(f"3. Isolated Dynamic Points (Removed Person):   {dynamic_only_points_count:,} points -> {dynamic_only_ply.name}")
    print(f"Dynamic Points Removed:                        {dynamic_points_removed:,} points ({reduction_pct:.2f}%)")
    print(f"Mask Overhead:                                 {mask_time/len(images)*1000:.1f} ms/frame")
    print(f"Debug Visuals Saved to:                        {vis_dir_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
