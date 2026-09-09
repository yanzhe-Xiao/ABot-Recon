#!/usr/bin/env python3
"""Method 1: KISS-Matcher coarse SE(3), then small_gicp VGICP refinement."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import kiss_matcher as km
import numpy as np
import small_gicp

from common import load_cloud_npz, matrix, registration_metrics, transform_points, write_ply


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--voxel", type=float, default=0.05); p.add_argument("--eval-threshold", type=float, default=0.15)
    return p.parse_args()


def save_transform(directory: Path, name: str, value: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", value)
    np.savetxt(directory / f"{name}.txt", value, fmt="%.10g")


def main() -> None:
    args = parse_args(); out = args.output_dir; transforms = out / "transforms"; clouds = out / "pointclouds"
    transforms.mkdir(parents=True, exist_ok=True); clouds.mkdir(parents=True, exist_ok=True)
    src, tgt = load_cloud_npz(args.source), load_cloud_npz(args.target)
    metrics: dict[str, object] = {"method": "KISS-Matcher -> small_gicp VGICP", "status": "started",
                                  "source_points": len(src["points"]), "target_points": len(tgt["points"]),
                                  "voxel": args.voxel, "before": registration_metrics(src["points"], tgt["points"], np.eye(4), 1.0, args.eval_threshold)}
    log = []
    try:
        start = time.perf_counter(); matcher = km.KISSMatcher(km.KISSMatcherConfig(args.voxel))
        result = matcher.estimate(src["points"].astype(np.float32), tgt["points"].astype(np.float32))
        coarse_runtime = time.perf_counter() - start
        coarse = matrix(np.asarray(result.rotation), np.asarray(result.translation))
        save_transform(transforms, "coarse_transform", coarse)
        coarse_metrics = registration_metrics(src["points"], tgt["points"], coarse, 1.0, args.eval_threshold)
        coarse_metrics.update({"runtime_sec": coarse_runtime,
                               "rotation_inliers": int(matcher.get_num_rotation_inliers()),
                               "final_inliers": int(matcher.get_num_final_inliers()),
                               "library_score": None})
        metrics["coarse"] = coarse_metrics
        coarse_ok = (coarse_metrics["fitness"] >= 0.08 and coarse_metrics["overlap_nn_median"] <= args.eval_threshold * 2.0
                     and coarse_metrics["final_inliers"] >= 5 and coarse_metrics["translation_norm"] <= 30.0)
        metrics["coarse_status"] = "success" if coarse_ok else "coarse_registration_failed"
        aligned = transform_points(src["points"], coarse)
        write_ply(clouds / "06_clean_coarse_in_08.ply", aligned, src["colors"], src["confidence"])
        if not coarse_ok:
            metrics.update({"refine_status": "fusion_not_attempted", "status": "coarse_registration_failed",
                            "runtime_total_sec": coarse_runtime})
        else:
            start = time.perf_counter()
            refined = small_gicp.align(tgt["points"].astype(np.float64), src["points"].astype(np.float64),
                init_T_target_source=coarse, registration_type="VGICP", voxel_resolution=args.voxel,
                downsampling_resolution=args.voxel, max_correspondence_distance=args.eval_threshold,
                num_threads=8, max_iterations=50, verbose=False)
            refine_runtime = time.perf_counter() - start
            fine = np.asarray(refined.T_target_source); save_transform(transforms, "fine_transform", fine)
            fine_metrics = registration_metrics(src["points"], tgt["points"], fine, 1.0, args.eval_threshold)
            fine_metrics.update({"runtime_sec": refine_runtime, "converged": bool(refined.converged),
                                 "iterations": int(refined.iterations), "library_num_inliers": int(refined.num_inliers),
                                 "library_error": float(refined.error)})
            metrics["fine"] = fine_metrics
            fine_ok = bool(refined.converged and fine_metrics["fitness"] >= max(0.08, coarse_metrics["fitness"] * 0.9)
                           and fine_metrics["overlap_nn_median"] <= coarse_metrics["overlap_nn_median"] * 1.15)
            metrics["refine_status"] = "success" if fine_ok else "gicp_failed"
            metrics["status"] = "likely_success" if fine_ok else "uncertain"
            metrics["runtime_total_sec"] = coarse_runtime + refine_runtime
            aligned = transform_points(src["points"], fine)
            write_ply(clouds / "06_clean_refined_in_08.ply", aligned, src["colors"], src["confidence"])
    except Exception:
        metrics["status"] = "dependency_failed" if "coarse" not in metrics else "gicp_failed"
        metrics["error"] = traceback.format_exc(); log.append(metrics["error"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "run.log").write_text("\n".join(log) + "\n" + json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
