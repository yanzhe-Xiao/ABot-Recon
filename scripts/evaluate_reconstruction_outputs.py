#!/usr/bin/env python3
"""Validate one ABot-Recon experiment and write smoke-friendly metrics/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def tensor(path: Path) -> torch.Tensor:
    return torch.as_tensor(torch.load(path, map_location="cpu", weights_only=True)).cpu()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"min": math.nan, "median": math.nan, "mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "min": float(np.min(values)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def ffprobe(path: Path) -> dict[str, object]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ], text=True)
    payload = json.loads(raw)
    stream = payload["streams"][0]
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "codec": stream.get("codec_name"), "width": int(stream["width"]),
        "height": int(stream["height"]), "fps": stream.get("avg_frame_rate"),
        "frame_count": int(stream.get("nb_read_frames", 0)),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def ply_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"No vertex count in {path}")


def main() -> None:
    cfg = args()
    out = cfg.output_dir
    benchmark = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    poses = np.asarray(np.load(out / "camera_poses.npy"), dtype=np.float64)
    relative = np.asarray(np.load(out / "relative_poses.npy"), dtype=np.float64)
    points = tensor(out / "local_points.pt").float()
    confidence = tensor(out / "confidence.pt").float()
    mask = tensor(out / "confidence_mask.pt").bool()
    colors = tensor(out / "colors.pt")

    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    identity = np.eye(3)[None]
    orth_error = np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - identity, axis=(1, 2))
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    rel_rotation = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    cos_angle = np.clip((np.trace(rel_rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angle))
    finite_points = torch.isfinite(points).all(dim=-1)
    finite_conf = torch.isfinite(confidence)

    required = [
        "camera_poses.npy", "relative_poses.npy", "local_points.pt", "confidence.pt",
        "confidence_mask.pt", "colors.pt", "metadata.json", "experiment_config.json",
        "pointcloud.ply", "overview.mp4", "accumulation.mp4", "run.log",
    ]
    artifact_checks = {name: (out / name).is_file() and (out / name).stat().st_size > 0 for name in required}
    shape_checks = {
        "camera_poses": list(poses.shape) == [120, 4, 4],
        "relative_poses": list(relative.shape) == [119, 4, 4],
        "local_points": list(points.shape) == [120, 280, 504, 3],
        "confidence": list(confidence.shape) == [120, 280, 504],
        "confidence_mask": list(mask.shape) == [120, 280, 504],
        "colors": list(colors.shape) == [120, 280, 504, 3],
    }
    numeric_checks = {
        "poses_all_finite": bool(np.isfinite(poses).all()),
        "relative_poses_all_finite": bool(np.isfinite(relative).all()),
        "colors_uint8": colors.dtype == torch.uint8,
        "confidence_in_unit_interval": bool(finite_conf.all() and confidence.min() >= 0 and confidence.max() <= 1),
        "rotation_matrices_valid": bool(orth_error.max() < 1e-3 and determinant_error.max() < 1e-3),
        "local_points_have_finite_values": bool(finite_points.any()),
    }
    videos = {name: ffprobe(out / name) for name in ("overview.mp4", "accumulation.mp4")}
    smoke_pass = all(artifact_checks.values()) and all(shape_checks.values()) and all(numeric_checks.values())
    payload = {
        "experiment_id": cfg.experiment_id, "video_id": cfg.video_id,
        "status": "passed" if smoke_pass else "failed", "smoke_pass": smoke_pass,
        "benchmark": benchmark,
        "artifacts": artifact_checks, "shapes": {
            "camera_poses": list(poses.shape), "relative_poses": list(relative.shape),
            "local_points": list(points.shape), "confidence": list(confidence.shape),
            "confidence_mask": list(mask.shape), "colors": list(colors.shape),
        },
        "checks": {**shape_checks, **numeric_checks},
        "trajectory": {
            "path_length": float(steps.sum()), "translation_step": stats(steps),
            "rotation_step_deg": stats(angles_deg),
            "rotation_orthogonality_error": stats(orth_error),
            "rotation_determinant_error": stats(determinant_error),
        },
        "points": {
            "finite_fraction": float(finite_points.float().mean()),
            "positive_local_z_fraction_of_finite": float((points[..., 2][finite_points] > 0).float().mean()),
            "local_z": stats(points[..., 2][finite_points].numpy()),
        },
        "confidence": {
            "finite_fraction": float(finite_conf.float().mean()),
            "distribution": stats(confidence[finite_conf].numpy()),
            "mask_true_fraction": float(mask.float().mean()),
        },
        "pointcloud": {"path": str(out / "pointcloud.ply"), "bytes": (out / "pointcloud.ply").stat().st_size, "vertices": ply_vertices(out / "pointcloud.ply")},
        "videos": videos,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    row = {
        "experiment_id": cfg.experiment_id, "video_id": cfg.video_id,
        "status": payload["status"], "frames": benchmark["frames"],
        "inference_seconds": benchmark["inference_seconds"], "fps": benchmark["fps"],
        "peak_cuda_vram_gib": benchmark["peak_cuda_vram_gib"],
        "path_length": payload["trajectory"]["path_length"],
        "finite_points_fraction": payload["points"]["finite_fraction"],
        "confidence_mean": payload["confidence"]["distribution"]["mean"],
        "ply_vertices": payload["pointcloud"]["vertices"],
    }
    cfg.csv.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if cfg.csv.exists():
        with cfg.csv.open(newline="", encoding="utf-8-sig") as handle:
            existing = [item for item in csv.DictReader(handle) if item.get("experiment_id") != cfg.experiment_id]
    with cfg.csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerows(existing); writer.writerow(row)
    print(json.dumps({"status": payload["status"], "checks": payload["checks"]}, indent=2))
    if not smoke_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
