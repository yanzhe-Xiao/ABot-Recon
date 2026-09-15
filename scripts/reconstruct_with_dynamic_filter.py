#!/usr/bin/env python3
"""
3D Point Cloud Reconstruction with Dynamic Object Removal (Scheme 1: 2D Semantic Masking).
Integrates YOLO-seg with ABot-Recon monocular streaming 3D reconstruction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image
from scripts.export_reconstruction_ply import write_binary_ply


class DynamicMaskGenerator:
    """Generates dynamic object masks using YOLO-seg with morphological dilation."""

    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        dynamic_classes: list[int] | None = None,
        conf_thresh: float = 0.25,
        dilate_kernel: int = 7,
        device: str = "cuda",
    ):
        self.model = YOLO(model_name)
        # Default: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 15: cat, 16: dog
        self.dynamic_classes = dynamic_classes if dynamic_classes is not None else [0, 1, 2, 3, 5, 7, 15, 16]
        self.conf_thresh = conf_thresh
        self.dilate_kernel = dilate_kernel
        self.device = device

    def predict_masks(self, image_paths: list[Path]) -> list[np.ndarray]:
        """
        Predict static boolean masks for a list of images.
        Returns: list of boolean ndarray with shape [H, W], True = Static, False = Dynamic (filtered).
        """
        static_masks = []
        for idx, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            results = self.model.predict(
                str(path),
                classes=self.dynamic_classes,
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
            dyn_mask = np.zeros((h, w), dtype=np.uint8)
            if len(results) > 0 and results[0].masks is not None:
                for mask_data in results[0].masks.data:
                    m = mask_data.cpu().numpy().astype(np.uint8)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    dyn_mask = np.bitwise_or(dyn_mask, m)

            if self.dilate_kernel > 0 and np.any(dyn_mask):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel)
                )
                dyn_mask = cv2.dilate(dyn_mask, kernel)

            static_mask = (dyn_mask == 0)
            static_masks.append(static_mask)
        return static_masks


def export_colored_pointcloud(
    world_points: torch.Tensor,
    colors: torch.Tensor,
    mask: torch.Tensor | None,
    output_ply: Path,
    point_stride: int = 2,
) -> int:
    """
    Subsamples and exports valid 3D points + RGB colors to a binary PLY file.
    world_points: [N, H, W, 3]
    colors: [N, H, W, 3] (uint8)
    mask: [N, H, W] (bool, True = valid)
    """
    pts = world_points[:, ::point_stride, ::point_stride, :].reshape(-1, 3)
    cls = colors[:, ::point_stride, ::point_stride, :].reshape(-1, 3)

    valid = torch.isfinite(pts).all(dim=-1)
    if mask is not None:
        sub_mask = mask[:, ::point_stride, ::point_stride].reshape(-1)
        valid = valid & sub_mask

    pts_valid = pts[valid].cpu().numpy().astype(np.float32)
    cls_valid = cls[valid].cpu().numpy().astype(np.uint8)
    empty_edges = np.empty((0, 2), dtype=np.int32)
    empty_edge_colors = np.empty((0, 3), dtype=np.uint8)
    write_binary_ply(output_ply, pts_valid, cls_valid, empty_edges, empty_edge_colors)
    return len(pts_valid)

def main():
    parser = argparse.ArgumentParser(description="ABot-Recon with Dynamic Mask Filtering")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/glasshouse_stride5/images"),
        help="Input frames directory",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints",
        help="Checkpoint directory or path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/glasshouse_dynamic_removal_scheme1"),
        help="Output directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--loop-closure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in args.image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")

    print(f"[1/4] Found {len(images)} images in {args.image_dir}")

    # 1. Compute 2D dynamic masks
    print("[2/4] Generating 2D dynamic masks with YOLO-seg...")
    mask_gen = DynamicMaskGenerator(
        model_name=args.yolo_model,
        conf_thresh=0.25,
        dilate_kernel=7,
        device=args.device,
    )
    t0 = time.time()
    static_masks_np = mask_gen.predict_masks(images)
    mask_time = time.time() - t0
    print(f"Mask generation finished in {mask_time:.2f}s ({len(images)/mask_time:.1f} FPS)")

    # 2. Run ABot-Recon Reconstruction
    print("[3/4] Running ABot-Recon 3D reconstruction...")
    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        attention_backend="auto",
        loop_closure=args.loop_closure,
        loop_salad_checkpoint=Path("checkpoints/loop/dino_salad.ckpt"),
        loop_dino_checkpoint=Path("checkpoints/loop/dinov2_vitb14_pretrain.pth"),
        loop_output_dir=args.output_dir / "loop",
        local_files_only=True,
    )

    t1 = time.time()
    result = model.infer(
        images,
        output_local_points=True,
        output_world_points=True,
        output_confidence=True,
        confidence_threshold=args.confidence_threshold,
        loop_closure=args.loop_closure,
    )
    recon_time = time.time() - t1
    print(f"ABot-Recon finished in {recon_time:.2f}s ({len(images)/recon_time:.1f} FPS)")

    # 3. Extract RGB colors
    colors_list = []
    for img_path in images:
        with Image.open(img_path) as img:
            tensor, _ = preprocess_image(img)
        colors_list.append((tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0))
    colors = torch.stack(colors_list)  # [N, H, W, 3]

    # Pre-processed shape matching
    H, W = result.local_points.shape[1], result.local_points.shape[2]
    resized_static_masks = []
    for m in static_masks_np:
        if m.shape != (H, W):
            m_res = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m_res = m
        resized_static_masks.append(torch.from_numpy(m_res))
    static_masks_tensor = torch.stack(resized_static_masks).to(result.local_points.device)  # [N, H, W]

    world_pts = result.world_points

    # 4. Export Both Baseline (without dynamic filter) and Filtered Pointclouds
    print("[4/4] Exporting Baseline vs Filtered 3D Point Clouds...")
    baseline_ply = args.output_dir / "reconstruction_baseline_with_dynamic.ply"
    filtered_ply = args.output_dir / "reconstruction_scheme1_filtered.ply"

    # Baseline mask: only confidence
    baseline_valid_mask = torch.isfinite(world_pts).all(dim=-1)
    if result.confidence_mask is not None:
        baseline_valid_mask = baseline_valid_mask & result.confidence_mask

    baseline_points_count = export_colored_pointcloud(
        world_pts, colors, baseline_valid_mask, baseline_ply, point_stride=args.point_stride
    )

    # Filtered mask: confidence & static mask
    filtered_valid_mask = baseline_valid_mask & static_masks_tensor
    filtered_points_count = export_colored_pointcloud(
        world_pts, colors, filtered_valid_mask, filtered_ply, point_stride=args.point_stride
    )

    dynamic_points_removed = baseline_points_count - filtered_points_count
    reduction_pct = (dynamic_points_removed / max(1, baseline_points_count)) * 100.0

    report = {
        "total_frames": len(images),
        "mask_generation_time_s": round(mask_time, 2),
        "recon_time_s": round(recon_time, 2),
        "baseline_points": baseline_points_count,
        "filtered_points": filtered_points_count,
        "dynamic_points_removed": dynamic_points_removed,
        "dynamic_removal_ratio_pct": round(reduction_pct, 2),
        "baseline_ply": str(baseline_ply),
        "filtered_ply": str(filtered_ply),
    }

    with (args.output_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("           Dynamic Object Removal Evaluation           ")
    print("=======================================================")
    print(f"Total Video Frames Processed: {len(images)}")
    print(f"Baseline Point Cloud (With Dynamic Objects): {baseline_points_count:,} points -> {baseline_ply.name}")
    print(f"Scheme 1 Point Cloud (Dynamic Filtered):     {filtered_points_count:,} points -> {filtered_ply.name}")
    print(f"Dynamic Points Removed:                     {dynamic_points_removed:,} points ({reduction_pct:.2f}%)")
    print(f"Mask Overhead:                              {mask_time/len(images)*1000:.1f} ms/frame")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
