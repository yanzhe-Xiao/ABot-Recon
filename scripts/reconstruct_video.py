#!/usr/bin/env python3
"""
3D Point Cloud Reconstruction from Video or Image Sequences using ABot-Recon.
Supports camera trajectory estimation, loop closure, dense point cloud generation,
and Open3D filtering/postprocessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from ultralytics import YOLO

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image
from scripts.export_reconstruction_ply import prepare_points, write_bev, write_binary_ply

def extract_frames_from_video(video_path: Path, frames_dir: Path, fps_target: float | None = None) -> list[Path]:
    """Extract frames from video into frames_dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(
        frames_dir.glob("*.jpg")
    )
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

def run_reconstruction(
    video_or_image_path: Path,
    output_dir: Path,
    device: str = "cuda",
    attention_backend: str = "auto",
    point_stride: int = 2,
    confidence_threshold: float = 0.2,
    loop_closure: bool = True,
    dynamic_filter: bool = False,
    dynamic_model: str = "yolo11m-seg.pt",
    dynamic_conf: float = 0.15,
    dynamic_dilate: int = 11,
    fps_target: float | None = None,
) -> dict:
    """Run full ABot-Recon reconstruction pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_output_dir = output_dir / "loop"
    loop_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine images
    if video_or_image_path.is_file():
        frames_dir = output_dir / "frames"
        image_paths = extract_frames_from_video(video_or_image_path, frames_dir, fps_target=fps_target)
    elif video_or_image_path.is_dir():
        image_paths = sorted(
            p for p in video_or_image_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {video_or_image_path}")

    if not image_paths:
        raise ValueError(f"No images found in {video_or_image_path}")

    print(f"\n{'='*70}")
    print(f"Starting 3D Reconstruction: {len(image_paths)} frames")
    print(f"Input:          {video_or_image_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Dynamic filter: {dynamic_filter} (model={dynamic_model}, conf={dynamic_conf}, dilate={dynamic_dilate})")
    print(f"Device:         {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'='*70}")
    # 2. Load model
    print("Loading ABot-Recon model checkpoints...")
    model = ABotRecon.from_pretrained(
        "checkpoints/abot_recon.safetensors",
        device=device,
        attention_backend=attention_backend,
        loop_closure=loop_closure,
        loop_salad_checkpoint="checkpoints/loop/dino_salad.ckpt" if loop_closure else None,
        loop_dino_checkpoint="checkpoints/loop/dinov2_vitb14_pretrain.pth" if loop_closure else None,
    )
    print("Model initialized.")

    # 3. Model inference
    t_start = time.time()
    print("Running streaming causal inference and pose estimation...")
    result = model.infer(
        image_paths,
        output_local_points=True,
        output_world_points=True,
        output_confidence=True,
        confidence_threshold=0.0,
        loop_closure=loop_closure,
    )
    t_infer = time.time() - t_start
    fps = len(image_paths) / t_infer
    print(f"Inference finished in {t_infer:.2f}s ({fps:.2f} fps, {1000/fps:.1f} ms/frame)")

    # 4. Save poses
    print("Saving camera poses...")
    poses_np = result.camera_poses.cpu().numpy()
    np.save(output_dir / "camera_poses.npy", poses_np)
    np.save(output_dir / "relative_poses.npy", result.relative_poses.cpu().numpy())
    np.save(output_dir / "camera_poses_noloop.npy", result.camera_poses_noloop.cpu().numpy())
    np.save(output_dir / "relative_poses_noloop.npy", result.relative_poses_noloop.cpu().numpy())
    if result.camera_poses_loop is not None:
        np.save(output_dir / "camera_poses_loop.npy", result.camera_poses_loop.cpu().numpy())
        np.save(output_dir / "relative_poses_loop.npy", result.relative_poses_loop.cpu().numpy())

    # 5. Save dense tensors
    print("Saving 3D point maps and confidence...")
    if result.local_points is not None:
        torch.save(result.local_points.cpu(), output_dir / "local_points.pt")
    if result.world_points is not None:
        torch.save(result.world_points.cpu(), output_dir / "world_points.pt")
    if result.confidence is not None:
        torch.save(result.confidence.cpu(), output_dir / "confidence.pt")
    if result.confidence_mask is not None:
        torch.save(result.confidence_mask.cpu(), output_dir / "confidence_mask.pt")

    # 6. Save colors & Preprocess frames for dynamic filter
    print("Saving preprocessed RGB colors...")
    colors = []
    preprocessed_rgbs = []
    for p in image_paths:
        with Image.open(p) as img:
            tensor, _ = preprocess_image(img)
        c_tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        colors.append(c_tensor)
        preprocessed_rgbs.append(c_tensor.numpy())
    colors_tensor = torch.stack(colors)
    torch.save(colors_tensor, output_dir / "colors.pt")

    # Optional Dynamic Object Removal Mask Generation
    static_masks_tensor = None
    t_mask = 0.0
    if dynamic_filter:
        print("\n[Dynamic Filter] Generating aligned 2D dynamic masks with YOLO-seg...")
        mask_gen = AlignedDynamicMaskGenerator(
            model_name=dynamic_model,
            conf_thresh=dynamic_conf,
            dilate_kernel=dynamic_dilate,
            device=device,
        )
        t_mask0 = time.time()
        static_masks_np = mask_gen.predict_masks_on_preprocessed(preprocessed_rgbs)
        t_mask = time.time() - t_mask0
        print(f"[Dynamic Filter] Generated {len(static_masks_np)} masks in {t_mask:.2f}s ({len(static_masks_np)/t_mask:.1f} FPS)")

        vis_dir = output_dir / "aligned_masks_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(image_paths) // 20)
        for idx in range(0, len(image_paths), step):
            rgb_vis = preprocessed_rgbs[idx].copy()
            dyn_pixels = ~static_masks_np[idx]
            rgb_vis[dyn_pixels] = [255, 0, 0]  # Mark dynamic pixels as red
            out_vis = cv2.addWeighted(preprocessed_rgbs[idx], 0.5, rgb_vis, 0.5, 0)
            cv2.imwrite(str(vis_dir / f"aligned_vis_{image_paths[idx].name}"), cv2.cvtColor(out_vis, cv2.COLOR_RGB2BGR))

        static_masks_tensor = torch.stack([torch.from_numpy(m) for m in static_masks_np])
        torch.save(static_masks_tensor, output_dir / "static_masks.pt")
    # 7. Save metadata
    metadata = dict(result.metadata)
    metadata["input_path"] = str(video_or_image_path)
    metadata["num_frames"] = len(image_paths)
    metadata["inference_time_s"] = round(t_infer, 2)
    metadata["fps"] = round(fps, 2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 8. Export PLY files
    print("\n--- Exporting Point Clouds ---")
    empty_edges = np.empty((0, 2), dtype=np.int32)
    empty_edge_colors = np.empty((0, 3), dtype=np.uint8)
    ply_standard = output_dir / "reconstruction.ply"

    world_pts_cpu = result.world_points.cpu()
    baseline_valid_mask = torch.isfinite(world_pts_cpu).all(dim=-1)
    if result.confidence is not None and confidence_threshold > 0:
        baseline_valid_mask = baseline_valid_mask & (result.confidence.cpu() >= confidence_threshold)

    eval_report = {}
    if dynamic_filter and static_masks_tensor is not None:
        baseline_ply = output_dir / "reconstruction_baseline_with_dynamic.ply"
        filtered_ply = output_dir / "reconstruction_scheme1_filtered.ply"
        dynamic_only_ply = output_dir / "reconstruction_removed_dynamic_only.ply"

        # Baseline PLY (unfiltered)
        baseline_points_count = export_colored_pointcloud(
            world_pts_cpu, colors_tensor, baseline_valid_mask, baseline_ply, point_stride=point_stride
        )
        print(f"Exported Baseline PLY: {baseline_ply} ({baseline_points_count:,} points)")

        # Filtered PLY (static map)
        filtered_valid_mask = baseline_valid_mask & static_masks_tensor
        filtered_points_count = export_colored_pointcloud(
            world_pts_cpu, colors_tensor, filtered_valid_mask, filtered_ply, point_stride=point_stride
        )
        shutil.copyfile(filtered_ply, ply_standard)
        print(f"Exported Filtered PLY: {ply_standard} ({filtered_points_count:,} points)")

        # Dynamic only PLY
        dynamic_only_mask = baseline_valid_mask & (~static_masks_tensor)
        dynamic_only_points_count = export_colored_pointcloud(
            world_pts_cpu, colors_tensor, dynamic_only_mask, dynamic_only_ply, point_stride=point_stride
        )
        print(f"Exported Dynamic Objects PLY: {dynamic_only_ply} ({dynamic_only_points_count:,} points)")

        dynamic_removed = baseline_points_count - filtered_points_count
        reduction_pct = (dynamic_removed / max(1, baseline_points_count)) * 100.0

        eval_report = {
            "total_frames": len(image_paths),
            "mask_generation_time_s": round(t_mask, 2),
            "recon_time_s": round(t_infer, 2),
            "baseline_points": baseline_points_count,
            "filtered_points": filtered_points_count,
            "dynamic_only_points": dynamic_only_points_count,
            "dynamic_points_removed": dynamic_removed,
            "dynamic_removal_ratio_pct": round(reduction_pct, 2),
            "baseline_ply": str(baseline_ply),
            "filtered_ply": str(filtered_ply),
            "dynamic_only_ply": str(dynamic_only_ply),
            "reconstruction_ply": str(ply_standard),
        }
        with (output_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)
    else:
        ply_args = argparse.Namespace(
            output=ply_standard,
            poses=output_dir / "camera_poses.npy",
            points=output_dir / "world_points.pt",
            colors=output_dir / "colors.pt",
            points_frame="world",
            metadata=output_dir / "metadata.json",
            confidence=output_dir / "confidence.pt",
            confidence_threshold=confidence_threshold,
            point_stride=point_stride,
            frame_stride=1,
            max_points=0,
            pose_stride=1,
            frustum_scale=0.15,
            bev_output=output_dir / "trajectory_bev.png",
            bev_size=1600,
            bev_plane="auto",
        )
        pts, pt_colors = prepare_points(ply_args, poses_np)
        write_binary_ply(ply_standard, pts, pt_colors, empty_edges, empty_edge_colors)
        print(f"Exported Standard PLY: {ply_standard} ({len(pts):,} points, {ply_standard.stat().st_size / 1024 / 1024:.2f} MB)")

    # BEV Trajectory map
    bev_png = output_dir / "trajectory_bev.png"
    plane = write_bev(bev_png, poses_np, 1600, "auto")
    print(f"Exported BEV Trajectory: {bev_png} ({len(poses_np)} poses on {plane.upper()} plane)")

    # Filtered & Denoised PLY via Open3D
    clean_ply_path = output_dir / "reconstruction_clean.ply"
    if ply_standard.exists() and ply_standard.stat().st_size > 0:
        print("Post-processing point cloud with Open3D statistical outlier removal...")
        pcd = o3d.io.read_point_cloud(str(ply_standard))
        if len(pcd.points) > 0:
            pcd_down = pcd.voxel_down_sample(voxel_size=0.015)
            cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
            clean_pcd = pcd_down.select_by_index(ind)
            o3d.io.write_point_cloud(str(clean_ply_path), clean_pcd, write_ascii=False)
            print(f"Exported Cleaned PLY: {clean_ply_path} ({len(clean_pcd.points):,} points, {clean_ply_path.stat().st_size / 1024 / 1024:.2f} MB)")
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Reconstruction completed successfully in {total_time:.2f}s!")
    print(f"{'='*70}\n")

    return {
        "num_frames": len(image_paths),
        "inference_time_s": t_infer,
        "total_time_s": total_time,
        "fps": fps,
        "ply_standard": str(ply_standard),
        "ply_clean": str(clean_ply_path),
        "bev_image": str(bev_png),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct 3D Point Cloud from Video or Images with Dynamic Filtering")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video or image directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--attention-backend", default="auto", choices=("auto", "paged", "sdpa"))
    parser.add_argument("--point-stride", type=int, default=2, help="Point sampling stride (1=densest, 2=high quality, 4=fast)")
    parser.add_argument("--confidence-threshold", type=float, default=0.2, help="Confidence filter threshold (0.0 to 1.0)")
    parser.add_argument("--no-loop-closure", action="store_true", help="Disable loop closure")
    parser.add_argument("--dynamic-filter", action=argparse.BooleanOptionalAction, default=False, help="Enable 2D YOLO dynamic object removal")
    parser.add_argument("--dynamic-model", default="yolo11m-seg.pt", help="YOLO segmentation model path")
    parser.add_argument("--dynamic-conf", type=float, default=0.15, help="Confidence threshold for dynamic detection")
    parser.add_argument("--dynamic-dilate", type=int, default=11, help="Dilation kernel size for dynamic masks")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for frame extraction from video")
    args = parser.parse_args()

    run_reconstruction(
        video_or_image_path=args.input,
        output_dir=args.output,
        device=args.device,
        attention_backend=args.attention_backend,
        point_stride=args.point_stride,
        confidence_threshold=args.confidence_threshold,
        loop_closure=not args.no_loop_closure,
        dynamic_filter=args.dynamic_filter,
        dynamic_model=args.dynamic_model,
        dynamic_conf=args.dynamic_conf,
        dynamic_dilate=args.dynamic_dilate,
        fps_target=args.fps,
    )

if __name__ == "__main__":
    main()
