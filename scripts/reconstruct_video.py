#!/usr/bin/env python3
"""
3D Point Cloud Reconstruction from Video or Image Sequences using ABot-Recon.
Supports camera trajectory estimation, loop closure, dense point cloud generation,
and Open3D filtering/postprocessing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import open3d as o3d
import torch
from PIL import Image

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


def run_reconstruction(
    video_or_image_path: Path,
    output_dir: Path,
    device: str = "cuda",
    attention_backend: str = "auto",
    point_stride: int = 2,
    confidence_threshold: float = 0.2,
    loop_closure: bool = True,
) -> dict:
    """Run full ABot-Recon reconstruction pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_output_dir = output_dir / "loop"
    loop_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine images
    if video_or_image_path.is_file():
        frames_dir = output_dir / "frames"
        image_paths = extract_frames_from_video(video_or_image_path, frames_dir)
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
    print(f"Input:      {video_or_image_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
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

    # 6. Save colors
    print("Saving preprocessed RGB colors...")
    colors = []
    for p in image_paths:
        with Image.open(p) as img:
            tensor, _ = preprocess_image(img)
        colors.append((tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0))
    colors_tensor = torch.stack(colors)
    torch.save(colors_tensor, output_dir / "colors.pt")

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

    # Standard PLY (point_stride=2 for balance between density and file size)
    ply_standard = output_dir / "reconstruction.ply"
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
    plane = write_bev(ply_args.bev_output, poses_np, ply_args.bev_size, ply_args.bev_plane)
    print(f"Exported BEV Trajectory: {ply_args.bev_output} ({len(poses_np)} poses on {plane.upper()} plane)")

    # Filtered & Denoised PLY via Open3D
    clean_ply_path = output_dir / "reconstruction_clean.ply"
    if len(pts) > 0:
        print("Post-processing point cloud with Open3D statistical outlier removal...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(pt_colors.astype(np.float64) / 255.0)

        # Voxel downsampling + statistical outlier removal
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
        "bev_image": str(ply_args.bev_output),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct 3D Point Cloud from Video or Images")
    parser.add_argument("--input", "-i", type=Path, default=Path("/home/data/xyz/ABot-Recon/data/玻璃房/玻璃房1.mp4"), help="Input video or image directory")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/玻璃房1"), help="Output directory")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--attention-backend", default="auto", choices=("auto", "paged", "sdpa"))
    parser.add_argument("--point-stride", type=int, default=2, help="Point sampling stride (1=densest, 2=high quality, 4=fast)")
    parser.add_argument("--confidence-threshold", type=float, default=0.2, help="Confidence filter threshold (0.0 to 1.0)")
    parser.add_argument("--no-loop-closure", action="store_true", help="Disable loop closure")
    args = parser.parse_args()

    run_reconstruction(
        video_or_image_path=args.input,
        output_dir=args.output,
        device=args.device,
        attention_backend=args.attention_backend,
        point_stride=args.point_stride,
        confidence_threshold=args.confidence_threshold,
        loop_closure=not args.no_loop_closure,
    )


if __name__ == "__main__":
    main()
