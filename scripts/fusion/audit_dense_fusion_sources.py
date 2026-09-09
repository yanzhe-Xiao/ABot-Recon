#!/usr/bin/env python3
"""Audit 06/08 participation in dense fusion and render source provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from common import write_ply
from render_dense_fusion_scene import (
    evenly_spaced_indices,
    read_abot_ply,
    transform_similarity,
    transform_source_poses,
    write_dense_orbit,
    write_path_video,
)


SOURCE_COLOR = np.array([255, 90, 35], dtype=np.uint8)
TARGET_COLOR = np.array([0, 205, 255], dtype=np.uint8)
MIXED_COLOR = np.array([80, 255, 100], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw-ply", type=Path, required=True)
    parser.add_argument("--target-raw-ply", type=Path, required=True)
    parser.add_argument("--source-poses", type=Path, required=True)
    parser.add_argument("--target-poses", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.015)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--focal-model-px", type=float, default=300.0)
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
    cloud_dir = args.output_dir / "pointclouds"
    render_dir = args.output_dir / "renders"
    cloud_dir.mkdir()
    render_dir.mkdir()

    source_points, source_colors, source_confidence = read_abot_ply(args.source_raw_ply)
    target_points, target_colors, target_confidence = read_abot_ply(args.target_raw_ply)
    transform = np.load(args.transform)
    source_points = transform_similarity(source_points, transform)
    source_valid = np.isfinite(source_points).all(axis=1) & np.isfinite(source_confidence)
    target_valid = np.isfinite(target_points).all(axis=1) & np.isfinite(target_confidence)
    source_valid &= source_confidence >= args.confidence_threshold
    target_valid &= target_confidence >= args.confidence_threshold
    source_points, source_colors, source_confidence = (
        source_points[source_valid], source_colors[source_valid], source_confidence[source_valid]
    )
    target_points, target_colors, target_confidence = (
        target_points[target_valid], target_colors[target_valid], target_confidence[target_valid]
    )

    source_count = len(source_points)
    points = np.concatenate((source_points, target_points))
    colors = np.concatenate((source_colors, target_colors))
    confidence = np.concatenate((source_confidence, target_confidence)).astype(np.float64)
    keys = np.floor(points / args.voxel).astype(np.int32)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel_count = len(counts)
    weights = np.maximum(confidence, 1e-6)
    sum_weight = np.bincount(inverse, weights=weights)
    fused_points = np.column_stack([
        np.bincount(inverse, weights=weights * points[:, axis]) for axis in range(3)
    ]) / sum_weight[:, None]
    fused_colors = np.column_stack([
        np.bincount(inverse, weights=weights * colors[:, axis]) for axis in range(3)
    ]) / sum_weight[:, None]
    fused_confidence = np.bincount(inverse, weights=weights * confidence) / sum_weight

    source_samples = np.bincount(inverse[:source_count], minlength=voxel_count)
    target_samples = np.bincount(inverse[source_count:], minlength=voxel_count)
    source_weight = np.bincount(inverse[:source_count], weights=weights[:source_count], minlength=voxel_count)
    target_weight = np.bincount(inverse[source_count:], weights=weights[source_count:], minlength=voxel_count)
    source_only = (source_samples > 0) & (target_samples == 0)
    target_only = (target_samples > 0) & (source_samples == 0)
    mixed = (source_samples > 0) & (target_samples > 0)
    source_fraction = source_weight / np.maximum(source_weight + target_weight, 1e-9)

    provenance_colors = np.empty((voxel_count, 3), dtype=np.uint8)
    provenance_colors[source_only] = SOURCE_COLOR
    provenance_colors[target_only] = TARGET_COLOR
    provenance_colors[mixed] = MIXED_COLOR
    weighted_rgb = np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8)
    fused_points = fused_points.astype(np.float32)
    fused_confidence = fused_confidence.astype(np.float32)
    write_ply(cloud_dir / "fused_confidence_weighted_rgb.ply", fused_points, weighted_rgb, fused_confidence)
    write_ply(cloud_dir / "fused_source_provenance.ply", fused_points, provenance_colors, source_fraction.astype(np.float32))
    np.savez_compressed(
        cloud_dir / "fused_with_source_provenance.npz",
        points=fused_points,
        colors=weighted_rgb,
        confidence=fused_confidence,
        source_samples=source_samples.astype(np.int32),
        target_samples=target_samples.astype(np.int32),
        source_weight_fraction=source_fraction.astype(np.float32),
    )

    source_poses = transform_source_poses(np.load(args.source_poses), transform)
    target_poses = np.load(args.target_poses).astype(np.float64)
    source_indices = evenly_spaced_indices(len(source_poses), args.frames)
    target_indices = evenly_spaced_indices(len(target_poses), args.frames)
    videos = [
        render_dir / "08_camera_path_weighted_rgb_fusion.mp4",
        render_dir / "08_camera_path_source_provenance.mp4",
        render_dir / "06_clean_camera_path_source_provenance.mp4",
        render_dir / "orbit_source_provenance.mp4",
    ]
    write_path_video(videos[0], fused_points, weighted_rgb, target_poses, target_indices,
                     args.focal_model_px, args.fps, "06+08 confidence-weighted RGB fusion",
                     "08 camera path")
    provenance_note = "orange=06 only | cyan=08 only | green=mixed voxel"
    write_path_video(videos[1], fused_points, provenance_colors, target_poses, target_indices,
                     args.focal_model_px, args.fps, "Fusion source provenance", provenance_note)
    write_path_video(videos[2], fused_points, provenance_colors, source_poses, source_indices,
                     args.focal_model_px, args.fps, "Fusion source provenance", provenance_note)
    write_dense_orbit(videos[3], fused_points, provenance_colors, args.fps, args.frames,
                      "Fusion source provenance", provenance_note)

    metrics = {
        "status": "success",
        "transform": str(args.transform),
        "transform_direction": "06_clean world -> 08 world (Sim3 + VGICP)",
        "voxel_m": args.voxel,
        "confidence_threshold": args.confidence_threshold,
        "input_samples": {
            "06_clean": int(source_count),
            "08": int(len(target_points)),
            "total": int(len(points)),
            "06_clean_fraction": float(source_count / len(points)),
        },
        "output_voxels": {
            "total": int(voxel_count),
            "source_only": int(source_only.sum()),
            "target_only": int(target_only.sum()),
            "mixed": int(mixed.sum()),
            "source_only_fraction": float(source_only.mean()),
            "target_only_fraction": float(target_only.mean()),
            "mixed_fraction": float(mixed.mean()),
            "containing_06_clean": int((source_samples > 0).sum()),
            "containing_08": int((target_samples > 0).sum()),
        },
        "confidence_weight": {
            "06_clean_fraction": float(source_weight.sum() / (source_weight.sum() + target_weight.sum())),
            "08_fraction": float(target_weight.sum() / (source_weight.sum() + target_weight.sum())),
            "mixed_voxel_source_fraction_median": float(np.median(source_fraction[mixed])) if mixed.any() else None,
        },
        "provenance_colors": {
            "orange": "06_clean only",
            "cyan": "08 only",
            "green": "voxel contains samples from both 06_clean and 08",
        },
        "videos": {video.name: probe(video) for video in videos},
    }
    (args.output_dir / "fusion_source_audit.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
