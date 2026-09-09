#!/usr/bin/env python3
"""
Reconstruct 3D Point Clouds from Video Frames (05, 06, 07, 08)
using ABot-Recon with Paged Attention and Loop Closure.
Exports full dense point maps, poses, metadata, and colored binary PLY.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from PIL import Image

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image
from scripts.export_reconstruction_ply import prepare_points, write_bev, write_binary_ply


def reconstruct_sequence(
    model: ABotRecon,
    seq_id: str,
    image_dir: Path,
    output_dir: Path,
    loop_output_dir: Path,
    point_stride: int = 4,
    device: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    print(f"\n{'='*70}")
    print(f"Reconstructing Sequence {seq_id} ({len(paths)} frames)...")
    print(f"Input dir:  {image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}")

    t_start = time.time()
    result = model.infer(
        paths,
        output_local_points=True,
        output_world_points=True,
        output_confidence=True,
        confidence_threshold=0.0,
        loop_closure=True,
    )
    t_infer = time.time() - t_start
    fps = len(paths) / t_infer
    print(f"Inference finished in {t_infer:.2f}s ({fps:.2f} fps, {1000/fps:.1f} ms/frame)")

    # 1. Save pose arrays
    print("Saving camera poses...")
    np.save(output_dir / "camera_poses.npy", result.camera_poses.cpu().numpy())
    np.save(output_dir / "relative_poses.npy", result.relative_poses.cpu().numpy())
    np.save(output_dir / "camera_poses_noloop.npy", result.camera_poses_noloop.cpu().numpy())
    np.save(output_dir / "relative_poses_noloop.npy", result.relative_poses_noloop.cpu().numpy())
    if result.camera_poses_loop is not None:
        np.save(output_dir / "camera_poses_loop.npy", result.camera_poses_loop.cpu().numpy())
        np.save(output_dir / "relative_poses_loop.npy", result.relative_poses_loop.cpu().numpy())

    # 2. Save dense tensors
    print("Saving point maps and confidence...")
    if result.local_points is not None:
        torch.save(result.local_points.cpu(), output_dir / "local_points.pt")
    if result.world_points is not None:
        torch.save(result.world_points.cpu(), output_dir / "world_points.pt")
    if result.confidence is not None:
        torch.save(result.confidence.cpu(), output_dir / "confidence.pt")
    if result.confidence_mask is not None:
        torch.save(result.confidence_mask.cpu(), output_dir / "confidence_mask.pt")

    # 3. Save preprocessed RGB colors
    print("Saving colors tensor...")
    colors = []
    for p in paths:
        with Image.open(p) as img:
            tensor, _ = preprocess_image(img)
        colors.append((tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0))
    torch.save(torch.stack(colors), output_dir / "colors.pt")

    # 4. Save metadata JSON
    metadata = dict(result.metadata)
    metadata["seq_id"] = seq_id
    metadata["image_dir"] = str(image_dir)
    metadata["inference_time_s"] = round(t_infer, 2)
    metadata["fps"] = round(fps, 2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5. Export colored binary PLY and BEV
    print(f"Exporting reconstruction.ply (point stride = {point_stride})...")
    ply_args = argparse.Namespace(
        output=output_dir / "reconstruction.ply",
        poses=output_dir / "camera_poses.npy",
        points=output_dir / "world_points.pt",
        colors=output_dir / "colors.pt",
        points_frame="world",
        metadata=output_dir / "metadata.json",
        confidence=output_dir / "confidence.pt",
        confidence_threshold=None,
        point_stride=point_stride,
        frame_stride=1,
        max_points=0,  # keep all points
        pose_stride=1,
        frustum_scale=0.15,
        bev_output=output_dir / "trajectory_bev.png",
        bev_size=1600,
        bev_plane="auto",
    )
    poses_np = result.camera_poses.cpu().numpy()
    pts, pt_colors = prepare_points(ply_args, poses_np)
    empty_edges = np.empty((0, 2), dtype=np.int32)
    empty_edge_colors = np.empty((0, 3), dtype=np.uint8)
    if hasattr(model, "reset"):
        model.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_binary_ply(ply_args.output, pts, pt_colors, empty_edges, empty_edge_colors)
    print(f"Wrote {ply_args.output}: {len(pts):,} RGB points ({ply_args.output.stat().st_size / 1024 / 1024:.1f} MB)")

    plane = write_bev(ply_args.bev_output, poses_np, ply_args.bev_size, ply_args.bev_plane)
    print(f"Wrote {ply_args.bev_output}: {len(poses_np)} poses on {plane.upper()} plane")
    print(f"Sequence {seq_id} completed in {time.time() - t_start:.2f}s!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct 05, 06, 07, 08 sequences")
    parser.add_argument("--sequences", nargs="+", default=["05", "06", "07", "08"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", default="paged")
    parser.add_argument("--point-stride", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading ABotRecon model on {args.device} with {args.attention_backend} backend...")
    model = ABotRecon.from_pretrained(
        "checkpoints/abot_recon.safetensors",
        device=args.device,
        attention_backend=args.attention_backend,
        loop_closure=True,
        loop_salad_checkpoint="checkpoints/loop/dino_salad.ckpt",
        loop_dino_checkpoint="checkpoints/loop/dinov2_vitb14_pretrain.pth",
    )
    print("Model initialized.")

    for seq_id in args.sequences:
        img_dir = Path(f"data/data/{seq_id}")
        out_dir = Path(f"outputs/data_{seq_id}_loop")
        loop_dir = out_dir / "loop"
        reconstruct_sequence(
            model=model,
            seq_id=seq_id,
            image_dir=img_dir,
            output_dir=out_dir,
            loop_output_dir=loop_dir,
            point_stride=args.point_stride,
            device=args.device,
        )

    print("\nAll requested sequences reconstructed successfully!")


if __name__ == "__main__":
    main()
