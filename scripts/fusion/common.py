"""Shared, dependency-light helpers for ABot cross-video registration."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def load_tensor(path: Path) -> torch.Tensor:
    return torch.as_tensor(torch.load(path, map_location="cpu", weights_only=True)).cpu()


def validate_baseline(directory: Path, expected_frames: int) -> dict[str, object]:
    required = ["camera_poses.npy", "relative_poses.npy", "local_points.pt",
                "confidence.pt", "confidence_mask.pt", "colors.pt", "benchmark.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {missing}")
    poses = np.load(directory / "camera_poses.npy", mmap_mode="r")
    relative = np.load(directory / "relative_poses.npy", mmap_mode="r")
    if poses.shape != (expected_frames, 4, 4) or relative.shape != (expected_frames - 1, 4, 4):
        raise ValueError(f"Unexpected pose shapes: {poses.shape}, {relative.shape}")
    if not np.isfinite(poses).all() or not np.isfinite(relative).all():
        raise ValueError("Pose outputs contain NaN/Inf")
    return json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))


def complete_baseline_metadata(directory: Path, video: Path, video_id: str, frames: int) -> None:
    metadata = directory / "metadata.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({
            "frames": frames, "loop_closure": False, "attention_backend": "sdpa",
            "dense_output_indices": list(range(frames)),
            "dense_outputs": {"local_points": True, "world_points": False, "confidence": True},
            "confidence_threshold": 0.0, "pose_outputs": ["noloop"],
            "benchmark_file": "benchmark.json",
            "note": "Recovered after a post-inference adapter metadata-write error; tensors were already complete."
        }, indent=2), encoding="utf-8")
    config = directory / "experiment_config.json"
    if not config.exists():
        benchmark = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
        config.write_text(json.dumps({
            "experiment_id": f"{video_id}_baseline", "status": "passed",
            "video_id": video_id, "video_path": str(video), "frames": frames,
            "stride": 1, "dense_stride": 1, "confidence_threshold": 0.0,
            "height": 280, "width": 504, "amp_dtype": "bf16",
            "attention_backend": "sdpa", "local_window_frames": 12,
            "rot_correction_kernel": 10, "rot_correction_max_deg": 2.0,
            "loop_closure": False, "benchmark": benchmark,
        }, indent=2), encoding="utf-8")


def load_cloud(directory: Path, *, confidence_threshold: float = 0.3,
               pixel_stride: int = 4, frame_stride: int = 1) -> dict[str, np.ndarray]:
    points = load_tensor(directory / "local_points.pt").float()
    confidence = load_tensor(directory / "confidence.pt").float()
    colors = load_tensor(directory / "colors.pt")
    poses = np.asarray(np.load(directory / "camera_poses.npy"), dtype=np.float64)
    chunks_p, chunks_c, chunks_w, chunks_f = [], [], [], []
    depths = []
    for frame in range(0, len(points), frame_stride):
        local = points[frame, ::pixel_stride, ::pixel_stride].numpy()
        conf = confidence[frame, ::pixel_stride, ::pixel_stride].numpy()
        rgb = colors[frame, ::pixel_stride, ::pixel_stride].numpy()
        valid = np.isfinite(local).all(axis=-1) & np.isfinite(conf) & (local[..., 2] > 0)
        valid &= conf >= confidence_threshold
        local = local[valid].astype(np.float64, copy=False)
        world = local @ poses[frame, :3, :3].T + poses[frame, :3, 3]
        chunks_p.append(world.astype(np.float32))
        chunks_c.append(rgb[valid].astype(np.uint8))
        chunks_w.append(conf[valid].astype(np.float32))
        chunks_f.append(np.full(len(world), frame, dtype=np.int32))
        depths.append(local[:, 2].astype(np.float32))
    return {
        "points": np.concatenate(chunks_p), "colors": np.concatenate(chunks_c),
        "confidence": np.concatenate(chunks_w), "frame": np.concatenate(chunks_f),
        "depth": np.concatenate(depths), "poses": poses,
    }


def voxel_fuse(points: np.ndarray, colors: np.ndarray, confidence: np.ndarray,
               voxel: float, confidence_weighted: bool) -> dict[str, np.ndarray | int]:
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    weights = np.maximum(confidence.astype(np.float64), 1e-6) if confidence_weighted else np.ones(len(points))
    sum_w = np.bincount(inverse, weights=weights)
    out_p = np.column_stack([np.bincount(inverse, weights=weights * points[:, j]) for j in range(3)]) / sum_w[:, None]
    out_c = np.column_stack([np.bincount(inverse, weights=weights * colors[:, j]) for j in range(3)]) / sum_w[:, None]
    out_w = np.bincount(inverse, weights=weights * confidence) / sum_w
    return {"points": out_p.astype(np.float32), "colors": np.clip(np.rint(out_c), 0, 255).astype(np.uint8),
            "confidence": out_w.astype(np.float32), "counts": counts.astype(np.int32),
            "duplicate_voxels": int((counts > 1).sum())}


def save_cloud_npz(path: Path, cloud: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=cloud["points"], colors=cloud["colors"], confidence=cloud["confidence"])


def load_cloud_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in ("points", "colors", "confidence")}


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_conf = confidence is not None
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue"]
    if has_conf:
        header.append("property float confidence")
    header.append("end_header")
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if has_conf:
        dtype.append(("confidence", "<f4"))
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    if has_conf:
        vertices["confidence"] = confidence
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii")); vertices.tofile(handle)


def transform_points(points: np.ndarray, transform: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return (scale * points.astype(np.float64)) @ transform[:3, :3].T + transform[:3, 3]


def rotation_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def registration_metrics(source: np.ndarray, target: np.ndarray, transform: np.ndarray,
                         scale: float, threshold: float, max_samples: int = 300_000) -> dict[str, float | int]:
    rng = np.random.default_rng(20260908)
    src = source if len(source) <= max_samples else source[rng.choice(len(source), max_samples, replace=False)]
    tgt = target if len(target) <= max_samples else target[rng.choice(len(target), max_samples, replace=False)]
    aligned = transform_points(src, transform, scale)
    distances = cKDTree(tgt).query(aligned, workers=-1)[0]
    inliers = distances <= threshold
    count = int(inliers.sum())
    return {
        "fitness": float(inliers.mean()), "inlier_rmse": float(np.sqrt(np.mean(distances[inliers] ** 2))) if count else float("inf"),
        "inlier_count": count, "inlier_ratio": float(inliers.mean()),
        "overlap_nn_median": float(np.median(distances)), "overlap_nn_p90": float(np.percentile(distances, 90)),
        "estimated_translation": transform[:3, 3].tolist(), "translation_norm": float(np.linalg.norm(transform[:3, 3])),
        "estimated_rotation_deg": rotation_degrees(transform[:3, :3]), "estimated_scale": float(scale),
        "evaluation_threshold": float(threshold), "evaluated_source_points": int(len(src)),
    }


def umeyama(source: np.ndarray, target: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean, tgt_mean = source.mean(0), target.mean(0)
    src_c, tgt_c = source - src_mean, target - tgt_mean
    covariance = tgt_c.T @ src_c / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() / np.mean(np.sum(src_c ** 2, axis=1))) if estimate_scale else 1.0
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4); value[:3, :3] = rotation; value[:3, 3] = translation
    return value
