#!/usr/bin/env python3
"""Create an RGB + accumulated 3D reconstruction video without OpenGL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ABot-Recon RGB frames beside its accumulated point cloud."
    )
    parser.add_argument("--images", type=Path, default=Path("examples/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=("overview", "accumulation"), default="overview",
        help="overview shows RGB beside 3D; accumulation renders the 3D view full-frame.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--source-fps", type=float, default=30.0,
        help="Source video FPS used only for the displayed timestamp (default: 30).",
    )
    parser.add_argument(
        "--points-per-frame", type=int, default=2500,
        help="Maximum points sampled from each input frame (default: 2500).",
    )
    parser.add_argument(
        "--max-points", type=int, default=120000,
        help="Maximum accumulated points drawn in any video frame (default: 120000).",
    )
    parser.add_argument("--point-size", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-confidence-mask", action="store_true",
        help="Ignore confidence_mask.pt even when it exists.",
    )
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, (list, tuple)):
        value = torch.stack([torch.as_tensor(item) for item in value])
    value = torch.as_tensor(value).cpu()
    while value.ndim > 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def image_paths(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = [path for path in directory.iterdir() if path.suffix.lower() in extensions]
    return sorted(paths, key=lambda path: path.name)


def validate_shapes(
    points: torch.Tensor,
    colors: torch.Tensor,
    poses: np.ndarray,
    masks: torch.Tensor | None,
    frame_count: int,
    dense_indices: list[int],
) -> None:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"local_points must be [N,H,W,3], got {tuple(points.shape)}")
    if colors.shape != points.shape:
        raise ValueError(f"colors shape {tuple(colors.shape)} != points shape {tuple(points.shape)}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"camera_poses must be [N,4,4], got {poses.shape}")
    if len(poses) != frame_count:
        raise ValueError(f"image/pose frame counts differ: {frame_count} != {len(poses)}")
    if len(colors) != len(points) or len(dense_indices) != len(points):
        raise ValueError(
            "dense point/color/index counts differ: "
            f"{len(points)}/{len(colors)}/{len(dense_indices)}"
        )
    if any(index < 0 or index >= frame_count for index in dense_indices):
        raise ValueError("metadata contains an out-of-range dense frame index")
    if masks is not None and tuple(masks.shape) != tuple(points.shape[:-1]):
        raise ValueError(f"confidence mask shape {tuple(masks.shape)} is incompatible")


def sample_world_points(
    local: np.ndarray,
    color: np.ndarray,
    pose: np.ndarray,
    mask: np.ndarray | None,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    flat_points = local.reshape(-1, 3)
    flat_colors = color.reshape(-1, 3)
    valid = np.isfinite(flat_points).all(axis=1)
    if mask is not None:
        valid &= mask.reshape(-1).astype(bool)
    indices = np.flatnonzero(valid)
    if limit > 0 and len(indices) > limit:
        indices = rng.choice(indices, size=limit, replace=False)
    selected = flat_points[indices].astype(np.float32, copy=False)
    world = selected @ pose[:3, :3].T + pose[:3, 3]
    rgb = flat_colors[indices]
    if np.issubdtype(rgb.dtype, np.integer) or (rgb.size and rgb.max() > 1.0):
        rgb = rgb.astype(np.float32) / 255.0
    return world, np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)


def equal_3d_limits(ax, low: np.ndarray, high: np.ndarray) -> None:
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.52, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output = args.output or output_dir / f"{args.mode}.mp4"
    paths = image_paths(args.images)
    if not paths:
        raise FileNotFoundError(f"no input images found in {args.images}")

    points = load_tensor(output_dir / "local_points.pt")
    colors = load_tensor(output_dir / "colors.pt")
    poses = np.load(output_dir / "camera_poses.npy").astype(np.float32)
    mask_path = output_dir / "confidence_mask.pt"
    masks = None if args.no_confidence_mask or not mask_path.exists() else load_tensor(mask_path)
    metadata_path = output_dir / "metadata.json"
    dense_indices = None
    if metadata_path.exists():
        dense_indices = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "dense_output_indices"
        )
    if dense_indices is None:
        dense_indices = list(range(len(points)))
    dense_indices = [int(index) for index in dense_indices]
    validate_shapes(points, colors, poses, masks, len(paths), dense_indices)

    rng = np.random.default_rng(args.seed)
    sampled_points: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    for dense_index, pose_index in enumerate(dense_indices):
        frame_mask = None if masks is None else masks[dense_index].numpy()
        world, rgb = sample_world_points(
            points[dense_index].numpy(), colors[dense_index].numpy(), poses[pose_index], frame_mask,
            args.points_per_frame, rng,
        )
        sampled_points.append(world)
        sampled_colors.append(rgb)

    nonempty = [item for item in sampled_points if len(item)]
    if not nonempty:
        raise ValueError("no valid 3D points remain after filtering")
    bounds_data = np.concatenate(nonempty + [poses[:, :3, 3]], axis=0)
    low, high = np.nanpercentile(bounds_data, [1.0, 99.0], axis=0)
    camera_xyz = poses[:, :3, 3]
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 7), dpi=args.dpi, facecolor="#0b0e14")
    if args.mode == "overview":
        grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.0), wspace=0.02)
        ax_image = fig.add_subplot(grid[0, 0])
        ax_cloud = fig.add_subplot(grid[0, 1], projection="3d")
    else:
        ax_image = None
        ax_cloud = fig.add_subplot(1, 1, 1, projection="3d")
    fig.suptitle(
        "ABot-Recon | accumulated point cloud + camera path",
        color="white", fontsize=16,
    )

    with imageio.get_writer(
        output, fps=args.fps, codec="libx264", quality=8,
        macro_block_size=2, ffmpeg_log_level="warning",
    ) as writer:
        accumulated_p: list[np.ndarray] = []
        accumulated_c: list[np.ndarray] = []
        dense_lookup = {pose_index: dense_index for dense_index, pose_index in enumerate(dense_indices)}
        for index, path in enumerate(paths):
            if index in dense_lookup:
                dense_index = dense_lookup[index]
                accumulated_p.append(sampled_points[dense_index])
                accumulated_c.append(sampled_colors[dense_index])
            cloud = np.concatenate(accumulated_p, axis=0)
            cloud_colors = np.concatenate(accumulated_c, axis=0)
            if args.max_points > 0 and len(cloud) > args.max_points:
                keep = np.linspace(0, len(cloud) - 1, args.max_points, dtype=np.int64)
                cloud, cloud_colors = cloud[keep], cloud_colors[keep]

            if ax_image is not None:
                ax_image.clear()
            ax_cloud.clear()
            timestamp = index / max(args.source_fps, 1e-9)
            if ax_image is not None:
                ax_image.imshow(Image.open(path).convert("RGB"))
                ax_image.set_title(
                    f"Input video  |  Frame {index + 1:03d}/{len(paths):03d}  |  {timestamp:06.2f} s",
                    color="white", loc="left",
                )
                ax_image.axis("off")

            ax_cloud.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud_colors,
                s=args.point_size, linewidths=0, depthshade=False, rasterized=True,
            )
            trajectory = camera_xyz[: index + 1]
            ax_cloud.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                          color="#27a8ff", linewidth=2.0, label="camera trajectory")
            ax_cloud.scatter(*camera_xyz[0], c="#37e36f", s=42, marker="o", label="start")
            ax_cloud.scatter(*camera_xyz[index], c="#ff3b45", s=75, marker="^", label="current camera")
            equal_3d_limits(ax_cloud, low, high)
            ax_cloud.view_init(elev=24, azim=-62)
            ax_cloud.set_xlabel("X", color="white")
            ax_cloud.set_ylabel("Y", color="white")
            ax_cloud.set_zlabel("Z", color="white")
            ax_cloud.set_title(
                f"Accumulated 3D point cloud  |  frame {index + 1:03d}/{len(paths):03d}"
                f"  |  {timestamp:06.2f} s  |  {len(cloud):,} points",
                color="white",
            )
            ax_cloud.set_facecolor("#0b0e14")
            ax_cloud.tick_params(colors="#c7cedb", labelsize=7)
            ax_cloud.grid(True, alpha=0.18)
            ax_cloud.legend(loc="upper right", fontsize=7, framealpha=0.75)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
            print(f"\rRendered {index + 1}/{len(paths)}", end="", flush=True)
    plt.close(fig)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
