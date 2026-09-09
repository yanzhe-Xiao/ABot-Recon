#!/usr/bin/env python3
"""Build a dense confidence fusion and render it from the recorded camera paths.

This renderer is deliberately CPU/headless.  It uses a depth-sorted RGB point
splat, so every rendered scene pixel comes from the fused point cloud; source
video frames are used only in the explicitly labelled comparison videos.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from common import write_ply


WIDTH, HEIGHT = 1280, 720
MODEL_WIDTH, MODEL_HEIGHT = 504, 280
BACKGROUND = np.array([18, 20, 24], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw-ply", type=Path, required=True)
    parser.add_argument("--target-raw-ply", type=Path, required=True)
    parser.add_argument("--source-poses", type=Path, required=True)
    parser.add_argument("--target-poses", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.015)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--focal-model-px", type=float, default=300.0)
    parser.add_argument("--max-render-points", type=int, default=1_200_000)
    return parser.parse_args()


def read_abot_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the binary PLY emitted by common.write_ply."""
    with path.open("rb") as handle:
        count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing end_header in {path}")
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            if line.strip() == b"end_header":
                offset = handle.tell()
                break
    if count is None:
        raise ValueError(f"Missing vertex count in {path}")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("confidence", "<f4"),
    ])
    data = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    colors = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    confidence = np.asarray(data["confidence"], dtype=np.float32).copy()
    return points, colors, confidence


def transform_similarity(points: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    return (points @ similarity[:3, :3].T + similarity[:3, 3]).astype(np.float32)


def dense_voxel_fuse(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel: float,
) -> dict[str, np.ndarray | int]:
    """Confidence-weight geometry and retain the sharpest representative RGB."""
    keys = np.floor(points / voxel).astype(np.int32)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    weights = np.maximum(confidence.astype(np.float64), 1e-6)
    sum_w = np.bincount(inverse, weights=weights)
    fused_points = np.column_stack([
        np.bincount(inverse, weights=weights * points[:, axis]) for axis in range(3)
    ]) / sum_w[:, None]
    fused_confidence = np.bincount(inverse, weights=weights * confidence) / sum_w

    # Averaging observations with different exposure makes indoor texture grey.
    # Choose the highest-confidence observation in each voxel for display RGB.
    order = np.lexsort((confidence, inverse))
    sorted_groups = inverse[order]
    last = np.r_[sorted_groups[1:] != sorted_groups[:-1], True]
    representatives = order[last]
    representative_groups = inverse[representatives]
    fused_colors = np.empty((len(counts), 3), dtype=np.uint8)
    fused_colors[representative_groups] = colors[representatives]
    return {
        "points": fused_points.astype(np.float32),
        "colors": fused_colors,
        "confidence": fused_confidence.astype(np.float32),
        "counts": counts.astype(np.int32),
        "duplicate_voxels": int((counts > 1).sum()),
    }


def proper_rotation(value: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(value)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def transform_source_poses(poses: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    scale = float(np.cbrt(np.linalg.det(similarity[:3, :3])))
    registration_rotation = proper_rotation(similarity[:3, :3] / scale)
    output = poses.copy().astype(np.float64)
    output[:, :3, 3] = poses[:, :3, 3] @ similarity[:3, :3].T + similarity[:3, 3]
    output[:, :3, :3] = np.einsum("ij,njk->nik", registration_rotation, poses[:, :3, :3])
    return output


def evenly_spaced_indices(count: int, frames: int) -> np.ndarray:
    return np.rint(np.linspace(0, count - 1, frames)).astype(np.int32)


def smooth_poses(poses: np.ndarray, radius: int = 2) -> np.ndarray:
    """Lightly smooth camera motion while preserving valid rotations."""
    if radius <= 0:
        return poses
    result = poses.copy()
    for index in range(len(poses)):
        lo, hi = max(0, index - radius), min(len(poses), index + radius + 1)
        result[index, :3, 3] = poses[lo:hi, :3, 3].mean(axis=0)
        result[index, :3, :3] = proper_rotation(poses[lo:hi, :3, :3].mean(axis=0))
    return result


def render_camera_view(
    points: np.ndarray,
    colors: np.ndarray,
    pose: np.ndarray,
    focal_model_px: float,
    *,
    point_radius: int = 2,
    max_depth: float = 12.0,
) -> np.ndarray:
    delta = points - pose[:3, 3]
    camera_points = delta @ pose[:3, :3]
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > 0.035) & (depth < max_depth)
    camera_points, depth, rgb = camera_points[valid], depth[valid], colors[valid]
    fx = focal_model_px * WIDTH / MODEL_WIDTH
    fy = focal_model_px * HEIGHT / MODEL_HEIGHT
    cx, cy = WIDTH * 0.5, HEIGHT * 0.5
    u = np.rint(fx * camera_points[:, 0] / depth + cx).astype(np.int32)
    v = np.rint(fy * camera_points[:, 1] / depth + cy).astype(np.int32)
    margin = point_radius + 1
    inside = (u >= margin) & (u < WIDTH - margin) & (v >= 48 + margin) & (v < HEIGHT - margin)
    u, v, depth, rgb = u[inside], v[inside], depth[inside], rgb[inside]
    order = np.argsort(depth)[::-1]
    u, v, rgb = u[order], v[order], rgb[order]
    canvas = np.broadcast_to(BACKGROUND, (HEIGHT, WIDTH, 3)).copy()
    offsets = [
        (du, dv)
        for dv in range(-point_radius, point_radius + 1)
        for du in range(-point_radius, point_radius + 1)
        if du * du + dv * dv <= point_radius * point_radius + 1
    ]
    for du, dv in offsets:
        canvas[v + dv, u + du] = rgb
    return canvas


def label(image: np.ndarray, text: str, subtitle: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (WIDTH, 48), (10, 12, 16), -1)
    cv2.putText(output, text, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, subtitle, (WIDTH - 460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 205, 255), 1, cv2.LINE_AA)
    return output


def load_selected_video_frames(video: Path, indices: np.ndarray) -> dict[int, np.ndarray]:
    wanted = set(int(index) for index in indices)
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    frame_index = 0
    while wanted:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            frames[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"Could not decode frames {sorted(wanted)} from {video}")
    return frames


def write_path_video(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    indices: np.ndarray,
    focal: float,
    fps: int,
    title: str,
    route: str,
    reference_frames: dict[int, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_poses = smooth_poses(poses[indices])
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                            macro_block_size=2, ffmpeg_log_level="warning") as writer:
        for output_index, (frame_index, pose) in enumerate(zip(indices, selected_poses)):
            rendered = render_camera_view(points, colors, pose, focal)
            rendered = label(rendered, title, f"{route} | frame {int(frame_index)}")
            if reference_frames is not None:
                reference = cv2.resize(reference_frames[int(frame_index)], (WIDTH // 2, HEIGHT))
                projected = cv2.resize(rendered, (WIDTH // 2, HEIGHT))
                comparison = np.concatenate((reference, projected), axis=1)
                cv2.putText(comparison, "Original video", (18, HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(comparison, "Dense fused cloud", (WIDTH // 2 + 18, HEIGHT - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                writer.append_data(comparison)
            else:
                writer.append_data(rendered)


def look_at_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    camera_down = np.array([0.0, 0.0, -1.0])
    right = np.cross(camera_down, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = position
    return pose


def write_dense_orbit(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    fps: int,
    frames: int,
    title: str = "Dense confidence fusion",
    subtitle: str = "1.5 cm voxel | max-confidence RGB",
) -> None:
    low, high = np.percentile(points, [1, 99], axis=0)
    center = (low + high) * 0.5
    extent = high - low
    horizontal_radius = max(float(np.linalg.norm(extent[:2])) * 0.70, 2.5)
    height = center[2] + max(float(extent[2]) * 0.15, 0.7)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                            macro_block_size=2, ffmpeg_log_level="warning") as writer:
        for index in range(frames):
            angle = 2 * np.pi * index / frames
            position = center + np.array([
                horizontal_radius * np.cos(angle), horizontal_radius * np.sin(angle), 0.0
            ])
            position[2] = height
            frame = render_camera_view(points, colors, look_at_pose(position, center), 390.0,
                                       point_radius=2, max_depth=30.0)
            writer.append_data(label(frame, title, subtitle))


def probe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cloud_dir = args.output_dir / "pointclouds"
    render_dir = args.output_dir / "renders"
    cloud_dir.mkdir()
    render_dir.mkdir()

    source_points, source_colors, source_confidence = read_abot_ply(args.source_raw_ply)
    target_points, target_colors, target_confidence = read_abot_ply(args.target_raw_ply)
    similarity = np.load(args.transform)
    source_points = transform_similarity(source_points, similarity)
    points = np.concatenate((source_points, target_points))
    colors = np.concatenate((source_colors, target_colors))
    confidence = np.concatenate((source_confidence, target_confidence))
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence >= args.confidence_threshold)
    points, colors, confidence = points[valid], colors[valid], confidence[valid]
    dense = dense_voxel_fuse(points, colors, confidence, args.voxel)
    np.savez_compressed(cloud_dir / "fused_confidence_dense.npz",
                        points=dense["points"], colors=dense["colors"], confidence=dense["confidence"])
    write_ply(cloud_dir / "fused_confidence_dense.ply", dense["points"], dense["colors"], dense["confidence"])

    render_points = dense["points"]
    render_colors = dense["colors"]
    if len(render_points) > args.max_render_points:
        rng = np.random.default_rng(20260908)
        chosen = rng.choice(len(render_points), args.max_render_points, replace=False)
        render_points, render_colors = render_points[chosen], render_colors[chosen]

    source_poses = transform_source_poses(np.load(args.source_poses), similarity)
    target_poses = np.load(args.target_poses).astype(np.float64)
    source_indices = evenly_spaced_indices(len(source_poses), args.frames)
    target_indices = evenly_spaced_indices(len(target_poses), args.frames)
    source_reference = load_selected_video_frames(args.source_video, source_indices)
    target_reference = load_selected_video_frames(args.target_video, target_indices)

    outputs = [
        render_dir / "06_clean_camera_path_fusion.mp4",
        render_dir / "08_camera_path_fusion.mp4",
        render_dir / "06_clean_input_vs_fusion.mp4",
        render_dir / "08_input_vs_fusion.mp4",
        render_dir / "dense_orbit.mp4",
    ]
    write_path_video(outputs[0], render_points, render_colors, source_poses, source_indices,
                     args.focal_model_px, args.fps, "Dense fused cloud", "06_clean camera path")
    write_path_video(outputs[1], render_points, render_colors, target_poses, target_indices,
                     args.focal_model_px, args.fps, "Dense fused cloud", "08 camera path")
    write_path_video(outputs[2], render_points, render_colors, source_poses, source_indices,
                     args.focal_model_px, args.fps, "Original vs fused projection", "06_clean",
                     source_reference)
    write_path_video(outputs[3], render_points, render_colors, target_poses, target_indices,
                     args.focal_model_px, args.fps, "Original vs fused projection", "08",
                     target_reference)
    write_dense_orbit(outputs[4], render_points, render_colors, args.fps, args.frames)

    still_indices = [30, 60, 90]
    for ordinal in still_indices:
        frame_index = int(target_indices[ordinal])
        still = render_camera_view(render_points, render_colors,
                                   smooth_poses(target_poses[target_indices])[ordinal],
                                   args.focal_model_px)
        imageio.imwrite(render_dir / f"08_camera_view_{ordinal:03d}_frame_{frame_index:04d}.png",
                       label(still, "Dense fused cloud", f"08 camera path | frame {frame_index}"))

    manifest = {
        "status": "success",
        "purpose": "recognizable scene visualization; registration transform unchanged",
        "source_cloud": str(args.source_raw_ply),
        "target_cloud": str(args.target_raw_ply),
        "transform": str(args.transform),
        "transform_direction": "06_clean world -> 08 world (Sim3 + VGICP)",
        "input_points": int(len(points)),
        "dense_fused_points": int(len(dense["points"])),
        "render_points": int(len(render_points)),
        "voxel_m": args.voxel,
        "confidence_threshold": args.confidence_threshold,
        "position_fusion": "confidence-weighted centroid, gamma=1",
        "display_color": "highest-confidence observation per voxel (avoids exposure averaging)",
        "renderer": "depth-sorted RGB point splat from recorded ABot camera poses",
        "point_splat_radius_px": 2,
        "focal_model_px": args.focal_model_px,
        "resolution": [WIDTH, HEIGHT],
        "fps": args.fps,
        "frames": args.frames,
        "comparison_note": "Original frames appear only in files named input_vs_fusion.",
        "videos": {path.name: probe(path) for path in outputs},
        "runtime_sec": time.perf_counter() - started,
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
