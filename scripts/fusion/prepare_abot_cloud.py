#!/usr/bin/env python3
"""Validate baselines, diagnose scale, and build registration clouds from tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (complete_baseline_metadata, load_cloud, save_cloud_npz,
                    validate_baseline, voxel_fuse, write_ply)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=0.05)
    return parser.parse_args()


def diagnostics(cloud: dict[str, np.ndarray]) -> dict[str, object]:
    poses, points, depth = cloud["poses"], cloud["points"], cloud["depth"]
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    low, high = np.percentile(points, [1, 99], axis=0)
    extent = high - low
    return {
        "camera_path_length": float(steps.sum()),
        "median_adjacent_translation": float(np.median(steps)),
        "p90_adjacent_translation": float(np.percentile(steps, 90)),
        "median_depth": float(np.median(depth)), "mean_depth": float(np.mean(depth)),
        "bbox_definition": "1st-to-99th percentile world-cloud extent after confidence/pixel filtering",
        "bbox_extent_x": float(extent[0]), "bbox_extent_y": float(extent[1]),
        "bbox_extent_z": float(extent[2]), "bbox_diagonal": float(np.linalg.norm(extent)),
        "sampled_points": int(len(points)),
    }


def main() -> None:
    args = parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    source_frames, target_frames = 855, 740
    src_bench = validate_baseline(args.source_dir, source_frames)
    tgt_bench = validate_baseline(args.target_dir, target_frames)
    complete_baseline_metadata(args.source_dir, args.source_video, "06_clean", source_frames)
    complete_baseline_metadata(args.target_dir, args.target_video, "08", target_frames)
    source = load_cloud(args.source_dir, confidence_threshold=args.confidence_threshold,
                        pixel_stride=args.pixel_stride)
    target = load_cloud(args.target_dir, confidence_threshold=args.confidence_threshold,
                        pixel_stride=args.pixel_stride)
    src_diag, tgt_diag = diagnostics(source), diagnostics(target)
    ratios = {key: src_diag[key] / tgt_diag[key] for key in (
        "camera_path_length", "median_adjacent_translation", "median_depth", "mean_depth", "bbox_diagonal")}
    central = float(np.median(list(ratios.values())))
    spread = float(max(ratios.values()) / min(ratios.values()))
    if 0.8 <= central <= 1.25 and spread <= 1.6:
        judgement = "A. 两段重建尺度基本一致"
    elif central < 0.7 or central > 1.43 or spread > 2.0:
        judgement = "B. 存在明显尺度差异"
    else:
        judgement = "C. 无法判断"
    payload = {"source": src_diag, "target": tgt_diag, "source_over_target_ratios": ratios,
               "ratio_median": central, "ratio_spread": spread, "judgement": judgement,
               "sampling": {"confidence_threshold": args.confidence_threshold,
                            "pixel_stride": args.pixel_stride, "frame_stride": 1,
                            "registration_voxel": args.voxel},
               "baseline_benchmarks": {"06_clean": src_bench, "08": tgt_bench}}
    (out / "scale_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, cloud in (("06_clean", source), ("08", target)):
        write_ply(out / f"{name}_raw.ply", cloud["points"], cloud["colors"], cloud["confidence"])
        fused = voxel_fuse(cloud["points"], cloud["colors"], cloud["confidence"], args.voxel, True)
        compact = {key: fused[key] for key in ("points", "colors", "confidence")}
        save_cloud_npz(out / f"{name}_voxel_{args.voxel:.2f}.npz", compact)
        write_ply(out / f"{name}_voxel_{args.voxel:.2f}.ply", compact["points"], compact["colors"], compact["confidence"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
