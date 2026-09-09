#!/usr/bin/env python3
"""Benchmark ABot-Recon streaming inference and save reconstruction outputs."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from abot_recon import ABotRecon
from abot_recon.geometry import relative_from_c2w
from abot_recon.preprocessing import iter_preprocessed, preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image-dir", type=Path)
    inputs.add_argument("--video", type=Path, help="Decode frames directly without writing PNGs")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="acvlab/ABot-Recon")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", default="sdpa", choices=("sdpa", "paged"))
    parser.add_argument("--amp-dtype", default="bf16", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--max-frames", type=int, default=22_000)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument(
        "--dense-stride", type=int, default=1,
        help="Run/save dense point and confidence heads every N frames; poses still use every frame.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    return parser.parse_args()


def collect_images(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return paths


def collect_video_frames(path: Path, start: int, count: int | None) -> list[torch.Tensor]:
    if start < 0 or (count is not None and count < 1):
        raise ValueError("--start-frame must be >= 0 and --num-frames must be >= 1")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[torch.Tensor] = []
    try:
        while count is None or len(frames) < count:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame, _ = preprocess_image(rgb)
            frames.append(frame.contiguous())
    finally:
        capture.release()
    if count is not None and len(frames) != count:
        raise ValueError(f"Requested {count} video frames, decoded {len(frames)}")
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def frames_only(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim >= 1 and value.shape[0] == 1:
        value = value[0]
    return value.detach().float().cpu()


def main() -> None:
    args = parse_args()
    if args.dense_stride < 1:
        raise ValueError("--dense-stride must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    error_path = args.output_dir / "benchmark_error.txt"
    if error_path.exists():
        error_path.unlink()

    if args.video is not None:
        # Keep decoded/preprocessed tensors in RAM: no multi-hundred-MiB PNG staging tree.
        cpu_frames = collect_video_frames(args.video, args.start_frame, args.num_frames)
        input_count = len(cpu_frames)
    else:
        paths = collect_images(args.image_dir)
        selected = paths[args.start_frame:]
        if args.num_frames is not None:
            selected = selected[:args.num_frames]
        # Decode and preprocess before model timing so I/O is excluded consistently.
        cpu_frames = [tensor.contiguous() for tensor, _ in iter_preprocessed(selected)]
        input_count = len(selected)
    height, width = cpu_frames[0].shape[-2:]

    try:
        recon = ABotRecon.from_pretrained(
            args.checkpoint,
            device=args.device,
            amp_dtype=args.amp_dtype,
            attention_backend=args.attention_backend,
            max_frames=args.max_frames,
            output_local_points=True,
            output_world_points=False,
            output_confidence=True,
            confidence_threshold=args.confidence_threshold,
            loop_closure=False,
        )
        runtime = recon.model
        dtype = runtime.compute_dtype

        def device_frames(source: list[torch.Tensor]):
            for tensor in source:
                yield tensor.unsqueeze(0).unsqueeze(0).to(
                    device=args.device, dtype=dtype, non_blocking=True
                )

        output_keys = ["camera_poses", "local_points", "conf"]
        dense_indices = list(range(0, len(cpu_frames), args.dense_stride))
        autocast_enabled = args.device.startswith("cuda") and dtype != torch.float32
        warmup_count = min(max(args.warmup_frames, 0), len(cpu_frames))
        if warmup_count:
            runtime.reset()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=dtype, enabled=autocast_enabled
            ):
                runtime.network.inference_stream_iter(
                    device_frames(cpu_frames[:warmup_count]),
                    num_frames=warmup_count,
                    output_keys=output_keys,
                )
            torch.cuda.synchronize()
            runtime.reset()
            torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=autocast_enabled
        ):
            raw = runtime.network.inference_stream_iter(
                device_frames(cpu_frames),
                num_frames=len(cpu_frames),
                output_keys=output_keys,
                dense_output_indices=dense_indices,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()

        # Everything below is outside the benchmark interval.
        poses = frames_only(raw["camera_poses"])
        local_points = frames_only(raw["local_points"])
        logits = frames_only(raw["conf"])
        if logits.ndim == 4 and logits.shape[-1] == 1:
            logits = logits[..., 0]
        confidence = torch.sigmoid(logits)
        confidence_mask = confidence >= args.confidence_threshold
        if args.confidence_threshold > 0:
            local_points = local_points.masked_fill(
                ~confidence_mask.unsqueeze(-1), float("nan")
            )
        relative = relative_from_c2w(poses)
        colors = torch.stack([
            (cpu_frames[index].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
            for index in dense_indices
        ])

        np.save(args.output_dir / "camera_poses.npy", poses.numpy())
        np.save(args.output_dir / "relative_poses.npy", relative.numpy())
        np.save(args.output_dir / "camera_poses_noloop.npy", poses.numpy())
        np.save(args.output_dir / "relative_poses_noloop.npy", relative.numpy())
        torch.save(local_points, args.output_dir / "local_points.pt")
        torch.save(colors, args.output_dir / "colors.pt")
        torch.save(confidence, args.output_dir / "confidence.pt")
        torch.save(confidence_mask, args.output_dir / "confidence_mask.pt")

        fps = input_count / elapsed
        report = {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "attention_backend_requested": args.attention_backend,
            "attention_backend_resolved": runtime.attention_backend,
            "amp_dtype": args.amp_dtype,
            "input_resolution": [width, height],
            "frames": input_count,
            "dense_stride": args.dense_stride,
            "dense_frames": len(dense_indices),
            "warmup_frames": warmup_count,
            "timing_scope": "preprocessed CPU tensors -> streaming model outputs; excludes load, decode, preprocess, CPU conversion and writes",
            "inference_seconds": elapsed,
            "fps": fps,
            "ms_per_frame": 1000.0 / fps,
            "peak_cuda_vram_bytes": peak_bytes,
            "peak_cuda_vram_gib": peak_bytes / (1024 ** 3),
            "p50_p95_latency": None,
            "latency_note": "The official streaming loop returns only after the sequence; per-frame timing would require instrumenting core loop behavior.",
            "loop_closure": False,
            "dense_outputs": ["local_points", "confidence"],
            "confidence_threshold": args.confidence_threshold,
        }
        (args.output_dir / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        metadata = {
            "frames": input_count,
            "loop_closure": False,
            "attention_backend": runtime.attention_backend,
            "dense_output_indices": dense_indices,
            "dense_outputs": {"local_points": True, "world_points": False, "confidence": True},
            "confidence_threshold": args.confidence_threshold,
            "pose_outputs": ["noloop"],
            "benchmark_file": "benchmark.json",
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except BaseException:
        error_text = traceback.format_exc()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            error_text += (
                f"\nCUDA memory after failure: free={free_bytes / 2**30:.3f} GiB, "
                f"total={total_bytes / 2**30:.3f} GiB\n"
            )
        except Exception:
            pass
        error_path.write_text(error_text, encoding="utf-8")
        print(error_text, flush=True)
        raise


if __name__ == "__main__":
    main()
