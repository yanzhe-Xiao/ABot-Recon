"""Online Frame-by-Frame Streaming 3D Reconstruction Engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from abot_recon import ABotRecon
from abot_recon.config import InferenceConfig
from abot_recon.preprocessing import preprocess_image


class OnlineReconstructionEngine:
    """Frame-by-frame causal online streaming 3D reconstruction engine.
    
    Processes individual video/camera frames with O(1) memory sliding window,
    yielding incremental 3D point cloud slices and real-time camera poses.
    """

    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/abot_recon.safetensors",
        device: str = "cuda",
        attention_backend: str = "auto",
        confidence_threshold: float = 0.1,
        point_stride: int = 4,
    ):
        self.config = InferenceConfig(
            checkpoint=Path(checkpoint),
            device=device,
            attention_backend=attention_backend,
            output_world_points=True,
            output_local_points=True,
            output_confidence=True,
            confidence_threshold=confidence_threshold,
            loop_closure=False,
        )
        self.model = ABotRecon.from_pretrained(
            self.config.checkpoint,
            device=self.config.device,
            attention_backend=self.config.attention_backend,
        )
        self.network = self.model.model.network
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.point_stride = max(1, int(point_stride))

        # Runtime streaming states
        self.frame_idx = 0
        self.ref_hidden: Optional[torch.Tensor] = None
        self.camera_state: Optional[dict[str, torch.Tensor]] = None
        self.paged_initialized = False

        # Global accumulated history (strictly identical to batch inference)
        self.history_poses: List[np.ndarray] = []
        self.history_points: List[torch.Tensor] = []
        self.history_colors: List[torch.Tensor] = []
        self.history_confidences: List[torch.Tensor] = []

        self.reset()

    def reset(self) -> None:
        """Reset streaming KV-cache and accumulated states."""
        self.frame_idx = 0
        self.ref_hidden = None
        self.camera_state = None
        self.history_poses.clear()
        self.history_points.clear()
        self.history_colors.clear()
        self.history_confidences.clear()
        self.model.reset()
        self.paged_initialized = False

    @torch.inference_mode()
    def process_frame(
        self,
        image_input: Image.Image | np.ndarray | str | Path | torch.Tensor,
        confidence_threshold: Optional[float] = None,
        point_stride: Optional[int] = None,
    ) -> dict[str, Any]:
        """Process a single incoming frame and return incremental 3D reconstruction."""
        start_time = time.perf_counter()

        # 1. Load and Preprocess Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = None

        if pil_img is not None:
            tensor_chw, _ = preprocess_image(pil_img, height=280, width=504)
        elif isinstance(image_input, torch.Tensor):
            tensor_chw = image_input.squeeze()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Frame tensor shape [1, 1, 3, H, W]
        frame = tensor_chw.unsqueeze(0).unsqueeze(0).to(self.device)
        if self.model.model.compute_dtype != torch.float32 and self.device.type == "cuda":
            frame = frame.to(self.model.model.compute_dtype)

        # 2. Lazy initialize paged manager on first frame
        if not self.paged_initialized and self.network._can_use_paged_kv(frame):
            B, _, _, H_img, W_img = frame.shape
            self.network._ensure_paged_manager(
                H=H_img, W=W_img, dtype=frame.dtype, device=frame.device
            )
            self.paged_initialized = True

        # 3. Causal Single-Frame Forward Step
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_enabled = (
            self.device.type == "cuda"
            and self.model.model.compute_dtype != torch.float32
        )

        with torch.autocast(
            device_type=device_type,
            dtype=self.model.model.compute_dtype,
            enabled=autocast_enabled,
        ):
            if self.paged_initialized:
                pred = self.network._forward_frame_paged(
                    frame,
                    frame_idx=self.frame_idx,
                    ref_hidden=self.ref_hidden,
                    camera_state=self.camera_state,
                )
            else:
                # Fallback path if paged attention is not available
                pred = self.network._forward_frame_sdpa(
                    frame,
                    frame_idx=self.frame_idx,
                    carry=None,
                    ref_hidden=self.ref_hidden,
                    camera_state=self.camera_state,
                )

        if pred.get("ref_hidden") is not None:
            self.ref_hidden = pred["ref_hidden"]
        self.camera_state = pred.get("camera_state", self.camera_state)

        # 4. Extract Camera Pose and World Points
        raw_pose = pred["camera_poses"][0, 0].detach().float().cpu().numpy()  # (4, 4)
        raw_world_pts = pred["points"][0, 0].detach().float().cpu()  # (280, 504, 3)

        conf_logits = pred.get("conf")
        if conf_logits is not None:
            if conf_logits.ndim == 5 and conf_logits.shape[-1] == 1:
                conf_logits = conf_logits[..., 0]
            conf_map = torch.sigmoid(conf_logits[0, 0].detach().float().cpu())  # (280, 504)
        else:
            conf_map = torch.ones((280, 504), dtype=torch.float32)

        # Store in historical state
        self.history_poses.append(raw_pose)
        self.history_points.append(raw_world_pts)
        self.history_confidences.append(conf_map)

        # 5. Extract Sampled Colored Points for Streaming Visualization
        stride = point_stride if point_stride is not None else self.point_stride
        thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        sampled_pts = raw_world_pts[::stride, ::stride].reshape(-1, 3).numpy()
        sampled_conf = conf_map[::stride, ::stride].reshape(-1).numpy()
        valid_mask = np.isfinite(sampled_pts).all(axis=-1)
        if thresh > 0:
            valid_mask &= (sampled_conf >= thresh)

        valid_pts = sampled_pts[valid_mask].astype(np.float32)

        # Extract RGB colors aligned with sampled pixels
        rgb_tensor = tensor_chw.permute(1, 2, 0)  # (280, 504, 3)
        sampled_rgb = (rgb_tensor[::stride, ::stride].reshape(-1, 3).numpy() * 255).astype(np.uint8)
        valid_colors = sampled_rgb[valid_mask]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        current_idx = self.frame_idx
        self.frame_idx += 1

        return {
            "frame_index": current_idx,
            "camera_pose": raw_pose.tolist(),  # 4x4 matrix as list
            "points_xyz": valid_pts,  # (M, 3) Float32
            "points_rgb": valid_colors,  # (M, 3) Uint8
            "point_count": len(valid_pts),
            "inference_time_ms": elapsed_ms,
            "fps": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
        }

    def stream_sequence(
        self,
        image_paths: Sequence[str | Path],
        confidence_threshold: Optional[float] = None,
        point_stride: Optional[int] = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream an entire image sequence frame-by-frame."""
        self.reset()
        for path in image_paths:
            yield self.process_frame(
                path,
                confidence_threshold=confidence_threshold,
                point_stride=point_stride,
            )
