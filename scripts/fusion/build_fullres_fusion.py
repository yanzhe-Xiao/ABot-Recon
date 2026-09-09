#!/usr/bin/env python3
"""Stream full 280x504 ABot point maps into a memory-bounded voxel fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import write_ply


PACK_BITS = 21
PACK_OFFSET = 1 << (PACK_BITS - 1)
PACK_MASK = (1 << PACK_BITS) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--chunk-frames", type=int, default=80)
    return parser.parse_args()


def pack_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    shifted = keys + PACK_OFFSET
    if shifted.min() < 0 or shifted.max() > PACK_MASK:
        raise ValueError("Point coordinate exceeds signed 21-bit packed voxel range")
    values = shifted.astype(np.uint64)
    return (values[:, 0] << np.uint64(42)) | (values[:, 1] << np.uint64(21)) | values[:, 2]


def reduce_sorted(
    codes: np.ndarray,
    sum_weight: np.ndarray,
    sum_points: np.ndarray,
    sum_colors: np.ndarray,
    sum_confidence: np.ndarray,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    return {
        "codes": sorted_codes[starts],
        "sum_weight": np.add.reduceat(sum_weight[order], starts),
        "sum_points": np.column_stack([
            np.add.reduceat(sum_points[order, axis], starts) for axis in range(3)
        ]),
        "sum_colors": np.column_stack([
            np.add.reduceat(sum_colors[order, axis], starts) for axis in range(3)
        ]),
        "sum_confidence": np.add.reduceat(sum_confidence[order], starts),
        "source_samples": np.add.reduceat(source_samples[order], starts),
        "target_samples": np.add.reduceat(target_samples[order], starts),
    }


def aggregate_raw(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel: float,
    source: bool,
) -> dict[str, np.ndarray]:
    weights = np.maximum(confidence.astype(np.float64), 1e-6)
    count = len(points)
    return reduce_sorted(
        pack_keys(points, voxel),
        weights,
        points.astype(np.float64) * weights[:, None],
        colors.astype(np.float64) * weights[:, None],
        confidence.astype(np.float64) * weights,
        np.ones(count, dtype=np.int64) if source else np.zeros(count, dtype=np.int64),
        np.zeros(count, dtype=np.int64) if source else np.ones(count, dtype=np.int64),
    )


def merge(left: dict[str, np.ndarray] | None, right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if left is None:
        return right
    return reduce_sorted(
        np.concatenate((left["codes"], right["codes"])),
        np.concatenate((left["sum_weight"], right["sum_weight"])),
        np.concatenate((left["sum_points"], right["sum_points"])),
        np.concatenate((left["sum_colors"], right["sum_colors"])),
        np.concatenate((left["sum_confidence"], right["sum_confidence"])),
        np.concatenate((left["source_samples"], right["source_samples"])),
        np.concatenate((left["target_samples"], right["target_samples"])),
    )


def transformed_pose(pose: np.ndarray, similarity: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = pose[:3, :3], pose[:3, 3]
    if similarity is None:
        return rotation, translation
    return similarity[:3, :3] @ rotation, similarity[:3, :3] @ translation + similarity[:3, 3]


def add_baseline(
    accumulator: dict[str, np.ndarray] | None,
    directory: Path,
    similarity: np.ndarray | None,
    voxel: float,
    threshold: float,
    chunk_frames: int,
    source: bool,
) -> tuple[dict[str, np.ndarray], int]:
    points = torch.load(directory / "local_points.pt", map_location="cpu", weights_only=True)
    confidence = torch.load(directory / "confidence.pt", map_location="cpu", weights_only=True)
    colors = torch.load(directory / "colors.pt", map_location="cpu", weights_only=True)
    poses = np.load(directory / "camera_poses.npy").astype(np.float64)
    accepted = 0
    for begin in range(0, len(points), chunk_frames):
        chunk_points: list[np.ndarray] = []
        chunk_colors: list[np.ndarray] = []
        chunk_confidence: list[np.ndarray] = []
        for frame in range(begin, min(begin + chunk_frames, len(points))):
            local = points[frame].float().numpy()
            conf = confidence[frame].float().numpy()
            rgb = colors[frame].numpy()
            valid = np.isfinite(local).all(axis=2) & np.isfinite(conf) & (local[..., 2] > 0)
            valid &= conf >= threshold
            rotation, translation = transformed_pose(poses[frame], similarity)
            world = local[valid].astype(np.float64) @ rotation.T + translation
            chunk_points.append(world.astype(np.float32))
            chunk_colors.append(rgb[valid].astype(np.uint8))
            chunk_confidence.append(conf[valid].astype(np.float32))
        raw_points = np.concatenate(chunk_points)
        raw_colors = np.concatenate(chunk_colors)
        raw_confidence = np.concatenate(chunk_confidence)
        accepted += len(raw_points)
        partial = aggregate_raw(raw_points, raw_colors, raw_confidence, voxel, source)
        accumulator = merge(accumulator, partial)
        print(json.dumps({
            "stream": "06_clean" if source else "08",
            "frames": [begin, min(begin + chunk_frames, len(points)) - 1],
            "accepted_points_so_far": accepted,
            "accumulated_voxels": len(accumulator["codes"]),
        }), flush=True)
    del points, confidence, colors
    assert accumulator is not None
    return accumulator, accepted


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    similarity = np.load(args.transform)
    accumulator = None
    accumulator, source_points = add_baseline(
        accumulator, args.source_dir, similarity, args.voxel,
        args.confidence_threshold, args.chunk_frames, True,
    )
    np.savez_compressed(args.output_dir / "source_accumulator_checkpoint.npz", **accumulator)
    accumulator, target_points = add_baseline(
        accumulator, args.target_dir, None, args.voxel,
        args.confidence_threshold, args.chunk_frames, False,
    )

    weight = accumulator["sum_weight"]
    points = (accumulator["sum_points"] / weight[:, None]).astype(np.float32)
    colors = np.clip(np.rint(accumulator["sum_colors"] / weight[:, None]), 0, 255).astype(np.uint8)
    confidence = (accumulator["sum_confidence"] / weight).astype(np.float32)
    source_samples = accumulator["source_samples"].astype(np.int32)
    target_samples = accumulator["target_samples"].astype(np.int32)
    source_fraction = (
        source_samples.astype(np.float64) /
        np.maximum(source_samples.astype(np.float64) + target_samples, 1.0)
    ).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "fused_fullres_confidence.npz",
        points=points,
        colors=colors,
        confidence=confidence,
        source_samples=source_samples,
        target_samples=target_samples,
        source_sample_fraction=source_fraction,
    )
    write_ply(args.output_dir / "fused_fullres_confidence.ply", points, colors, confidence)
    source_only = (source_samples > 0) & (target_samples == 0)
    target_only = (target_samples > 0) & (source_samples == 0)
    mixed = (source_samples > 0) & (target_samples > 0)
    metrics = {
        "status": "success",
        "sampling": "full 280x504 point map, pixel_stride=1, every frame",
        "voxel_m": args.voxel,
        "confidence_threshold": args.confidence_threshold,
        "input_points": {
            "06_clean": source_points,
            "08": target_points,
            "total": source_points + target_points,
        },
        "output_voxels": {
            "total": int(len(points)),
            "source_only": int(source_only.sum()),
            "target_only": int(target_only.sum()),
            "mixed": int(mixed.sum()),
            "containing_source": int((source_samples > 0).sum()),
        },
        "position_and_rgb": "confidence-weighted, gamma=1",
        "transform": str(args.transform),
        "runtime_sec": time.perf_counter() - started,
    }
    (args.output_dir / "fullres_fusion_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
