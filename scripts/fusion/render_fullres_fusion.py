#!/usr/bin/env python3
"""Render the full-resolution 06+08 fusion with depth-adaptive surfel splats."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from render_dense_fusion_scene import (
    evenly_spaced_indices,
    load_selected_video_frames,
    transform_source_poses,
    write_path_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--source-poses", type=Path, required=True)
    parser.add_argument("--target-poses", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--focal-model-px", type=float, default=300.0)
    parser.add_argument("--max-rgb-points", type=int, default=2_500_000)
    return parser.parse_args()


def probe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(args.cloud)
    points, colors = data["points"], data["colors"]
    source_samples, target_samples = data["source_samples"], data["target_samples"]
    rng = np.random.default_rng(20260908)
    if len(points) > args.max_rgb_points:
        rgb_keep = rng.choice(len(points), args.max_rgb_points, replace=False)
    else:
        rgb_keep = np.arange(len(points))
    rgb_points, rgb_colors = points[rgb_keep], colors[rgb_keep]

    source_voxels = np.flatnonzero(source_samples > 0)
    target_only = np.flatnonzero((target_samples > 0) & (source_samples == 0))
    target_budget = max(args.max_rgb_points - len(source_voxels), 0)
    target_keep = target_only if len(target_only) <= target_budget else rng.choice(
        target_only, target_budget, replace=False
    )
    provenance_keep = np.concatenate((source_voxels, target_keep))
    source_only_mask = (source_samples[provenance_keep] > 0) & (target_samples[provenance_keep] == 0)
    target_only_mask = (target_samples[provenance_keep] > 0) & (source_samples[provenance_keep] == 0)
    mixed_mask = (source_samples[provenance_keep] > 0) & (target_samples[provenance_keep] > 0)
    provenance_colors = np.empty((len(provenance_keep), 3), dtype=np.uint8)
    provenance_colors[source_only_mask] = np.array([255, 90, 35], dtype=np.uint8)
    provenance_colors[target_only_mask] = np.array([0, 205, 255], dtype=np.uint8)
    provenance_colors[mixed_mask] = np.array([80, 255, 100], dtype=np.uint8)
    provenance_points = points[provenance_keep]

    transform = np.load(args.transform)
    source_poses = transform_source_poses(np.load(args.source_poses), transform)
    target_poses = np.load(args.target_poses).astype(np.float64)
    source_indices = evenly_spaced_indices(len(source_poses), args.frames)
    target_indices = evenly_spaced_indices(len(target_poses), args.frames)
    target_frames = load_selected_video_frames(args.target_video, target_indices)

    videos = [
        args.output_dir / "08_camera_path_fullres_fusion.mp4",
        args.output_dir / "06_clean_camera_path_fullres_fusion.mp4",
        args.output_dir / "08_input_vs_fullres_fusion.mp4",
        args.output_dir / "08_camera_path_fullres_provenance.mp4",
    ]
    common = {"focal": args.focal_model_px, "fps": args.fps,
              "adaptive_voxel": args.voxel}
    write_path_video(videos[0], rgb_points, rgb_colors, target_poses, target_indices,
                     title="Full-resolution 06+08 fusion", route="08 camera path", **common)
    write_path_video(videos[1], rgb_points, rgb_colors, source_poses, source_indices,
                     title="Full-resolution 06+08 fusion", route="06_clean camera path", **common)
    write_path_video(videos[2], rgb_points, rgb_colors, target_poses, target_indices,
                     title="Original vs full-resolution fusion", route="08",
                     reference_frames=target_frames, **common)
    write_path_video(videos[3], provenance_points, provenance_colors, target_poses, target_indices,
                     title="Full-resolution fusion provenance",
                     route="orange=06 only | cyan=08 only | green=mixed", **common)

    manifest = {
        "status": "success",
        "cloud": str(args.cloud),
        "full_cloud_points": int(len(points)),
        "rgb_render_points": int(len(rgb_points)),
        "provenance_render_points": int(len(provenance_points)),
        "all_source_voxels_retained_in_provenance_render": True,
        "voxel_m": args.voxel,
        "renderer": "recorded camera path, z-buffered depth-adaptive surfel splat",
        "adaptive_radius_px": [1, 5],
        "videos": {path.name: probe(path) for path in videos},
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
