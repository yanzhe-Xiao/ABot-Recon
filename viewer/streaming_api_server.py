#!/usr/bin/env python3
"""
Real-time Video Stream 3D Reconstruction API Server
Supports persistent WebSocket and HTTP REST streaming with multi-client concurrency.

Key Features:
  1. Multi-Client Concurrency & Session Isolation:
     - Independent persistent sessions identified by unique `session_id`.
     - Shared model weights on GPU with isolated per-session KV caches and recurrent states.
  2. Configurable Real-time Parameters:
     - `point_stride`: spatial subsampling stride for streamed 3D points.
     - `frame_stride`: temporal subsampling stride for incoming video frames.
     - `voxel_size`: spatial voxel de-duplication grid size (meters) for final point cloud.
     - `confidence_threshold`: filter out low-confidence / noisy 3D points.
  3. Bidirectional Streaming:
     - Client streams raw binary JPEG/PNG frames or JSON base64.
     - Server responds with incremental 3D points, camera pose matrix [4,4], FPS, and latency.
  4. Defined End-of-Stream (EOS) Protocols:
     - Text message: `{"type": "EOS"}` or `{"action": "end"}` or `{"command": "finish"}`
     - Binary token: `b"EOS\x00\x00"`
     - Graceful WebSocket closure (code 1000)
     - REST endpoint: `POST /api/stream/end?session_id=<id>`
     - On EOS: finalizes point cloud, applies voxel de-duplication, and exports binary PLY.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import shutil
import sys
import time
import base64
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import open3d as o3d
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class AlignedDynamicMaskGenerator:
    """
    Generates dynamic object masks directly on preprocessed [H, W, 3] tensors,
    ensuring 100% pixel-perfect alignment with ABot-Recon pointmaps in real-time.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        dynamic_classes: list[int] | None = None,
        conf_thresh: float = 0.15,
        dilate_kernel: int = 7,
        device: str = "cuda:0",
    ):
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics package is required for dynamic object filtering. Please install it via pip install ultralytics")
        print(f"[Dynamic Filter] Initializing YOLO segmentation model '{model_name}' on {device}...")
        self.model = YOLO(model_name)
        self.dynamic_classes = dynamic_classes if dynamic_classes is not None else [0, 1, 2, 3, 5, 7, 15, 16]
        self.conf_thresh = conf_thresh
        self.dilate_kernel = dilate_kernel
        self.device = device

    def predict_single(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        rgb_uint8: [H, W, 3] uint8 RGB array (280 x 504)
        Returns: [H, W] boolean ndarray (True = Static / Keep, False = Dynamic / Filter out)
        """
        h, w = rgb_uint8.shape[:2]
        results = self.model.predict(
            rgb_uint8,
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
                    import cv2
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                dyn_mask = np.bitwise_or(dyn_mask, m)

        if self.dilate_kernel > 0 and np.any(dyn_mask):
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel))
            dyn_mask = cv2.dilate(dyn_mask, kernel)

        return dyn_mask == 0
# ---------------------------------------------------------------------------
# Session Data Structure
# ---------------------------------------------------------------------------
@dataclass
class StreamSession:
    session_id: str
    scene_id: str = ""
    robot: str = ""
    folder_name: str = ""
    point_stride: int = 4
    frame_stride: int = 1
    voxel_size: float = 0.015
    confidence_threshold: float = 0.1
    save_ply: bool = True
    save_trajectory: bool = True
    auto_fuse: bool = True
    dynamic_filter: bool = False
    frame_counter: int = 0  # Total frames received
    processed_counter: int = 0  # Frames actually forwarded through model
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    is_active: bool = True

    # Model recurrent streaming states
    ref_hidden: Optional[torch.Tensor] = None
    camera_state: Optional[Dict[str, torch.Tensor]] = None
    paged_manager: Any = None  # Session-specific FlashInfer PagedKVCacheManager

    # Global accumulated history
    history_poses: List[np.ndarray] = field(default_factory=list)
    history_points: List[torch.Tensor] = field(default_factory=list)
    history_colors: List[torch.Tensor] = field(default_factory=list)
    history_conf: List[torch.Tensor] = field(default_factory=list)
    history_static_masks: List[torch.Tensor] = field(default_factory=list)
    sampled_history_points: List[np.ndarray] = field(default_factory=list)
    sampled_history_colors: List[np.ndarray] = field(default_factory=list)
    latest_thumbnail: Optional[str] = None
    def __post_init__(self):
        if not self.scene_id:
            self.scene_id = self.session_id
        if not self.robot:
            import re
            m = re.search(r"(robot_[a-z0-9]+|robot[a-z0-9]+|cam_[a-z0-9]+|camera_[a-z0-9]+)", self.session_id, re.IGNORECASE)
            if m:
                self.robot = m.group(1)
            else:
                self.robot = "robot"
        if not self.folder_name:
            t_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.created_at))
            clean_scene = self.scene_id.replace(" ", "_")
            clean_robot = self.robot.replace(" ", "_")
            self.folder_name = f"{clean_scene}-{t_str}-{clean_robot}"
# ---------------------------------------------------------------------------
# Multi-Client Session Manager
# ---------------------------------------------------------------------------
class MultiSessionReconstructionManager:
    """Thread-safe and async-safe concurrent multi-stream session coordinator."""

    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/abot_recon.safetensors",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        output_base_dir: Path = REPO_ROOT / "outputs/streams",
        scenes_base_dir: Path = REPO_ROOT / "outputs/scenes",
        dynamic_filter: bool = True,
        dynamic_model: str = "yolo11m-seg.pt",
        dynamic_conf: float = 0.12,
        dynamic_dilate: int = 15,
    ):
        self.device = torch.device(device)
        self.output_base_dir = output_base_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_base_dir = scenes_base_dir
        self.scenes_base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, StreamSession] = {}
        self.subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.gpu_lock = asyncio.Lock()
        self.meta_lock = asyncio.Lock()
        self.scene_locks: Dict[str, asyncio.Lock] = {}
        self.scene_dirty: Dict[str, bool] = {}
        self.dynamic_filter = dynamic_filter
        self.dynamic_model = dynamic_model
        self.dynamic_conf = dynamic_conf
        self.dynamic_dilate = dynamic_dilate
        self.mask_generator: Optional[AlignedDynamicMaskGenerator] = None
        if self.dynamic_filter:
            self.mask_generator = AlignedDynamicMaskGenerator(
                model_name=self.dynamic_model,
                conf_thresh=self.dynamic_conf,
                dilate_kernel=self.dynamic_dilate,
                device=str(self.device),
            )
        print(f"[Server Engine] Loading ABot-Recon model on {self.device}...")
        self.model = ABotRecon.from_pretrained(
            checkpoint,
            device=str(self.device),
            attention_backend="paged",
        )
        self.network = self.model.model.network
        self.compute_dtype = self.model.model.compute_dtype

        # Template paged manager for resolution 280x504
        self.network._ensure_paged_manager(
            H=280,
            W=504,
            dtype=self.compute_dtype if self.compute_dtype != torch.float32 else torch.bfloat16,
            device=self.device,
        )
        self.paged_template = self.network._paged_manager
        print("[Server Engine] Model initialized and paged KV cache template ready.")

    def add_subscriber(self, session_id: str, queue: asyncio.Queue) -> None:
        """Register a real-time observer queue for an active session or '*' for all sessions."""
        self.subscribers[session_id].add(queue)

    def remove_subscriber(self, session_id: str, queue: asyncio.Queue) -> None:
        """Unregister a real-time observer queue."""
        if session_id in self.subscribers and queue in self.subscribers[session_id]:
            self.subscribers[session_id].remove(queue)
            if not self.subscribers[session_id]:
                del self.subscribers[session_id]

    def broadcast_to_subscribers(self, session_id: str, message: Dict[str, Any]) -> None:
        """Broadcast an event payload to all observers subscribed to this session or '*'."""
        targets = set()
        if session_id in self.subscribers:
            targets.update(self.subscribers[session_id])
        if "*" in self.subscribers:
            targets.update(self.subscribers["*"])
        for q in targets:
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                q.put_nowait(message)
            except Exception:
                pass

    def get_session_snapshot(self, session_id: str, max_points: int = 150000) -> Optional[Dict[str, Any]]:
        """Extract accumulated history snapshot for late-joining observers."""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        if not session.sampled_history_points:
            return None

        try:
            all_pts = np.concatenate(session.sampled_history_points, axis=0)
            all_rgb = np.concatenate(session.sampled_history_colors, axis=0)
        except Exception:
            return None

        total_pts = len(all_pts)
        if total_pts > max_points:
            indices = np.linspace(0, total_pts - 1, max_points, dtype=int)
            all_pts = all_pts[indices]
            all_rgb = all_rgb[indices]

        # Round coordinates to 4 decimals to reduce JSON payload size
        all_pts_rounded = np.round(all_pts, 4)

        poses_list = [p.tolist() if isinstance(p, np.ndarray) else p for p in session.history_poses]
        latest_pose = poses_list[-1] if poses_list else None

        return {
            "type": "history_sync",
            "session_id": session.session_id,
            "scene_id": session.scene_id,
            "frame_counter": session.frame_counter,
            "processed_counter": session.processed_counter,
            "poses": poses_list,
            "latest_pose": latest_pose,
            "points_xyz": all_pts_rounded.tolist(),
            "points_rgb": all_rgb.tolist(),
            "total_points_in_session": total_pts,
            "snapshot_point_count": len(all_pts),
            "latest_thumbnail": session.latest_thumbnail,
            "message": f"已恢复历史建图快照：共包含前 {session.processed_counter} 帧位姿轨迹与 {len(all_pts):,} 个已建点",
        }

    def get_scene_metadata_path(self, scene_id: str) -> Path:
        return self.scenes_base_dir / scene_id / "scene_metadata.json"

    def get_scene_metadata(self, scene_id: str) -> Dict[str, Any]:
        path = self.get_scene_metadata_path(scene_id)
        meta = None
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
        now = time.time()
        if meta is None:
            meta = {
                "scene_id": scene_id,
                "created_at": now,
                "updated_at": now,
                "completed_sessions": [],
                "active_sessions": [],
                "fusion_status": "none",
                "is_fused": False,
                "total_points": 0,
                "models": {},
            }
        # Dynamic disk sync to ensure zero-loss under high concurrency / simultaneous EOS
        completed = set(meta.get("completed_sessions", []))
        # Filter out symlinks or bare session_ids that alias across scenes
        cleaned = set()
        for s in completed:
            sp = self.output_base_dir / s
            if sp.is_symlink():
                continue
            cleaned.add(s)
        completed = cleaned

        if self.output_base_dir.is_dir():
            for d in self.output_base_dir.iterdir():
                if d.is_dir() and not d.is_symlink() and (d / "reconstruction.ply").is_file():
                    sum_file = d / "session_summary.json"
                    if sum_file.is_file():
                        try:
                            with open(sum_file, "r", encoding="utf-8") as sf:
                                sdata = json.load(sf)
                                if sdata.get("scene_id") == scene_id:
                                    completed.add(d.name)
                        except Exception:
                            pass
                    elif d.name.startswith(f"{scene_id}-") or d.name.startswith(f"{scene_id}_") or d.name == scene_id:
                        completed.add(d.name)
        meta["completed_sessions"] = sorted(list(completed))
        # Dynamic sync: keep active_sessions aligned with live in-memory sessions
        live_active = [sid for sid, s in self.sessions.items() if s.scene_id == scene_id]
        merged_active = set(live_active) | set(meta.get("active_sessions", []))
        meta["active_sessions"] = sorted(list(merged_active - set(meta["completed_sessions"])))
        return meta

    def save_scene_metadata(self, scene_id: str, meta: Dict[str, Any]) -> None:
        scene_dir = self.scenes_base_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        meta["updated_at"] = time.time()
        with open(self.get_scene_metadata_path(scene_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def register_session_to_scene(self, session_id: str, scene_id: str) -> None:
        meta = self.get_scene_metadata(scene_id)
        if session_id not in meta["active_sessions"] and session_id not in meta["completed_sessions"]:
            meta["active_sessions"].append(session_id)
            self.save_scene_metadata(scene_id, meta)

    async def schedule_scene_fusion(
        self,
        scene_id: str,
        voxel_size: float = 0.015,
        debounce_seconds: float = 0.5,
        force_full: bool = False,
    ) -> Dict[str, Any]:
        """Thread-safe and async-safe scene fusion coalescer for concurrent/simultaneous EOS."""
        if scene_id not in self.scene_locks:
            self.scene_locks[scene_id] = asyncio.Lock()

        scene_lock = self.scene_locks[scene_id]
        if scene_lock.locked():
            self.scene_dirty[scene_id] = True
            # Wait for active fusion to complete
            async with scene_lock:
                if not self.scene_dirty.get(scene_id, False):
                    # Already updated by earlier fusion pass
                    return self.get_scene_metadata(scene_id)
                self.scene_dirty[scene_id] = False
                return await self.fuse_scene(scene_id, voxel_size=voxel_size, force_full=force_full)

        async with scene_lock:
            if debounce_seconds > 0:
                await asyncio.sleep(debounce_seconds)
            self.scene_dirty[scene_id] = False
            return await self.fuse_scene(scene_id, voxel_size=voxel_size, force_full=force_full)

    def get_or_create_session(
        self,
        session_id: str,
        scene_id: Optional[str] = None,
        robot: Optional[str] = None,
        point_stride: int = 4,
        frame_stride: int = 1,
        voxel_size: float = 0.015,
        confidence_threshold: float = 0.1,
        save_ply: bool = True,
        save_trajectory: bool = True,
        auto_fuse: bool = True,
        dynamic_filter: Optional[bool] = None,
    ) -> StreamSession:
        effective_scene_id = scene_id.strip() if scene_id and scene_id.strip() else session_id
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.scene_id = effective_scene_id
            if robot:
                session.robot = robot.strip()
            session.point_stride = point_stride
            session.frame_stride = frame_stride
            session.voxel_size = voxel_size
            session.confidence_threshold = confidence_threshold
            session.auto_fuse = auto_fuse
            if dynamic_filter is not None:
                session.dynamic_filter = bool(dynamic_filter)
            session.last_active = time.time()
            self.register_session_to_scene(session_id, effective_scene_id)
            return session

        # Create fresh isolated paged manager
        paged_mgr = copy.deepcopy(self.paged_template)
        paged_mgr.reset()

        session = StreamSession(
            session_id=session_id,
            scene_id=effective_scene_id,
            robot=robot.strip() if robot else "",
            point_stride=max(1, int(point_stride)),
            frame_stride=max(1, int(frame_stride)),
            voxel_size=max(0.0, float(voxel_size)),
            confidence_threshold=max(0.0, min(1.0, float(confidence_threshold))),
            save_ply=save_ply,
            save_trajectory=save_trajectory,
            auto_fuse=auto_fuse,
            dynamic_filter=self.dynamic_filter if dynamic_filter is None else bool(dynamic_filter),
            paged_manager=paged_mgr,
        )
        self.sessions[session_id] = session
        self.register_session_to_scene(session_id, effective_scene_id)
        self.broadcast_to_subscribers("*", {
            "type": "session_created",
            "session_id": session_id,
            "scene_id": effective_scene_id,
            "robot": session.robot,
            "folder_name": session.folder_name,
        })
        print(f"[Session] Created new session [{session_id}] for scene [{effective_scene_id}] (folder: {session.folder_name}, active: {len(self.sessions)})")
        return session

    async def process_frame(
        self,
        session: StreamSession,
        image_bytes_or_pil: bytes | Image.Image | np.ndarray,
        include_point_data: bool = True,
    ) -> Dict[str, Any]:
        """Process one video frame for a specific session."""
        session.last_active = time.time()
        session.frame_counter += 1

        # Frame skip by frame_stride
        if (session.frame_counter - 1) % session.frame_stride != 0:
            return {
                "type": "frame_skipped",
                "session_id": session.session_id,
                "frame_counter": session.frame_counter,
                "reason": f"Skipped by frame_stride={session.frame_stride}",
            }

        t0 = time.perf_counter()

        # 1. Image Preprocessing (CPU, parallelizable)
        if isinstance(image_bytes_or_pil, bytes):
            pil_img = Image.open(io.BytesIO(image_bytes_or_pil)).convert("RGB")
        elif isinstance(image_bytes_or_pil, np.ndarray):
            pil_img = Image.fromarray(image_bytes_or_pil).convert("RGB")
        elif isinstance(image_bytes_or_pil, Image.Image):
            pil_img = image_bytes_or_pil.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_bytes_or_pil)}")

        tensor_chw, _ = preprocess_image(pil_img, height=280, width=504)
        frame = tensor_chw.unsqueeze(0).unsqueeze(0).to(self.device)
        if self.compute_dtype != torch.float32 and self.device.type == "cuda":
            frame = frame.to(self.compute_dtype)

        # 2. Forward Inference (Protected by GPU lock for multi-session safety)
        async with self.gpu_lock:
            # Swap in this session's isolated KV cache manager
            self.network._paged_manager = session.paged_manager

            with torch.inference_mode(), torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.compute_dtype,
                enabled=(self.compute_dtype != torch.float32),
            ):
                pred = self.network._forward_frame_paged(
                    frame,
                    frame_idx=session.processed_counter,
                    ref_hidden=session.ref_hidden,
                    camera_state=session.camera_state,
                )

            # Update session recurrent states
            if pred.get("ref_hidden") is not None:
                session.ref_hidden = pred["ref_hidden"]
            session.camera_state = pred.get("camera_state", session.camera_state)

            # Extract outputs
            raw_pose = pred["camera_poses"][0, 0].detach().float().cpu().numpy()
            raw_world = pred["points"][0, 0].detach().float().cpu()  # (280, 504, 3)

            conf_logits = pred.get("conf")
            if conf_logits is not None:
                if conf_logits.ndim == 5 and conf_logits.shape[-1] == 1:
                    conf_logits = conf_logits[..., 0]
                conf_map = torch.sigmoid(conf_logits[0, 0].detach().float().cpu())
            else:
                conf_map = torch.ones((280, 504), dtype=torch.float32)

        # 3. Postprocess Slice
        current_idx = session.processed_counter
        session.processed_counter += 1

        session.history_poses.append(raw_pose)
        session.history_points.append(raw_world)
        session.history_conf.append(conf_map)
        session.history_colors.append(
            (tensor_chw.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8)
        )

        static_mask = None
        if session.dynamic_filter:
            if self.mask_generator is None:
                self.mask_generator = AlignedDynamicMaskGenerator(
                    model_name=self.dynamic_model,
                    conf_thresh=self.dynamic_conf,
                    dilate_kernel=self.dynamic_dilate,
                    device=str(self.device),
                )
            rgb_full = (tensor_chw.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            static_mask = self.mask_generator.predict_single(rgb_full)
            session.history_static_masks.append(torch.from_numpy(static_mask))

        stride = session.point_stride
        thresh = session.confidence_threshold

        sampled_pts = raw_world[::stride, ::stride].reshape(-1, 3).numpy()
        sampled_conf = conf_map[::stride, ::stride].reshape(-1).numpy()
        valid = np.isfinite(sampled_pts).all(axis=-1)
        if thresh > 0:
            valid &= sampled_conf >= thresh
        if static_mask is not None:
            sampled_static = static_mask[::stride, ::stride].reshape(-1)
            valid &= sampled_static

        valid_pts = sampled_pts[valid].astype(np.float32)
        rgb_arr = (tensor_chw.permute(1, 2, 0)[::stride, ::stride].reshape(-1, 3).numpy() * 255).astype(np.uint8)
        valid_rgb = rgb_arr[valid]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_pts_approx = session.processed_counter * len(valid_pts)

        result: Dict[str, Any] = {
            "type": "frame_result",
            "session_id": session.session_id,
            "scene_id": session.scene_id,
            "frame_idx": current_idx,
            "frame_counter": session.frame_counter,
            "camera_pose": raw_pose.tolist(),  # 4x4 matrix
            "incremental_point_count": len(valid_pts),
            "total_points_approx": total_pts_approx,
            "inference_time_ms": round(elapsed_ms, 2),
            "fps": round(1000.0 / elapsed_ms, 2) if elapsed_ms > 0 else 0.0,
        }
        if include_point_data:
            result["points_xyz"] = valid_pts.tolist()
            result["points_rgb"] = valid_rgb.tolist()

        # Generate lightweight JPEG thumbnail for live HUD visualization
        thumb_b64 = None
        try:
            thumb_io = io.BytesIO()
            pil_img.resize((280, 157), Image.Resampling.BILINEAR).save(thumb_io, format="JPEG", quality=55)
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_io.getvalue()).decode("ascii")
            result["thumbnail"] = thumb_b64
        except Exception:
            result["thumbnail"] = None

        # Store in session sampled history for late-joining snapshot catchup
        session.sampled_history_points.append(valid_pts)
        session.sampled_history_colors.append(valid_rgb)
        session.latest_thumbnail = thumb_b64

        # Real-time broadcast to all attached viewers (always ensure 3D points are included for 3D viewers)
        broadcast_payload = dict(result)
        broadcast_payload["points_xyz"] = valid_pts.tolist()
        broadcast_payload["points_rgb"] = valid_rgb.tolist()
        broadcast_payload["thumbnail"] = thumb_b64
        self.broadcast_to_subscribers(session.session_id, broadcast_payload)
        return result

    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize a session, export PLY model, save trajectory, and clean up resources."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' does not exist")

        session = self.sessions[session_id]
        session.is_active = False
        t0 = time.time()
        session_dir = self.output_base_dir / session.folder_name
        session_dir.mkdir(parents=True, exist_ok=True)
        # Backward-compatible symlink if session_id != folder_name
        if session.session_id != session.folder_name:
            legacy_link = self.output_base_dir / session.session_id
            try:
                if legacy_link.is_symlink() or legacy_link.exists():
                    legacy_link.unlink()
                legacy_link.symlink_to(session.folder_name)
            except Exception:
                pass

        total_frames = len(session.history_points)
        print(f"\n[Finalize] Finalizing session [{session_id}] ({total_frames} frames)...")

        deliverables: Dict[str, Any] = {}
        dedup_count = 0
        raw_count = 0

        if total_frames > 0 and session.save_ply:
            # 1. Stack all point maps and colors
            all_pts = torch.stack(session.history_points)  # (N, H, W, 3)
            all_colors = torch.stack(session.history_colors)  # (N, H, W, 3)
            all_conf = torch.stack(session.history_conf)  # (N, H, W)
            # Save raw tensors for downstream Method 2 multimodal matching and fusion
            torch.save(all_pts.cpu(), session_dir / "world_points.pt")
            torch.save(all_colors.cpu(), session_dir / "colors.pt")
            torch.save(all_conf.cpu(), session_dir / "confidence.pt")
            deliverables["world_points_path"] = str(session_dir / "world_points.pt")
            deliverables["colors_path"] = str(session_dir / "colors.pt")
            deliverables["confidence_path"] = str(session_dir / "confidence.pt")
            if len(session.history_static_masks) == total_frames:
                all_masks = torch.stack(session.history_static_masks)
                torch.save(all_masks.cpu(), session_dir / "static_masks.pt")
                deliverables["static_masks_path"] = str(session_dir / "static_masks.pt")
            stride = session.point_stride
            thresh = session.confidence_threshold

            flat_pts = all_pts[:, ::stride, ::stride].reshape(-1, 3).numpy()
            flat_rgb = all_colors[:, ::stride, ::stride].reshape(-1, 3).numpy()
            flat_conf = all_conf[:, ::stride, ::stride].reshape(-1).numpy()

            valid = np.isfinite(flat_pts).all(axis=-1)
            if thresh > 0:
                valid &= flat_conf >= thresh
            if len(session.history_static_masks) == total_frames:
                flat_masks = torch.stack(session.history_static_masks)[:, ::stride, ::stride].reshape(-1).numpy()
                valid &= flat_masks
            pts_valid = flat_pts[valid].astype(np.float32)
            rgb_valid = flat_rgb[valid].astype(np.uint8)
            raw_count = len(pts_valid)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector((rgb_valid / 255.0).astype(np.float64))

            # Apply spatial voxel de-duplication if configured
            if session.voxel_size > 0:
                pcd_filtered = pcd.voxel_down_sample(session.voxel_size)
                dedup_count = len(pcd_filtered.points)
                final_pcd = pcd_filtered
            else:
                final_pcd = pcd
                dedup_count = raw_count

            ply_path = session_dir / "reconstruction.ply"
            if len(final_pcd.points) > 0:
                o3d.io.write_point_cloud(str(ply_path), final_pcd)
            else:
                with open(ply_path, "w", encoding="ascii") as f:
                    f.write("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            deliverables["ply_path"] = str(ply_path)
            deliverables["ply_url"] = f"/api/streams/{session_id}/reconstruction.ply"
            deliverables["ply_size_mb"] = round(ply_path.stat().st_size / 1024 / 1024, 2) if ply_path.is_file() else 0.0

        # 2. Save Trajectory
        if session.save_trajectory and len(session.history_poses) > 0:
            poses_np = np.stack(session.history_poses)
            pose_path = session_dir / "camera_poses.npy"
            np.save(pose_path, poses_np)
            deliverables["trajectory_path"] = str(pose_path)

        # 3. Update Scene Association & Barrier Tracking
        scene_id = session.scene_id
        # Eject from active in-memory session roster
        self.sessions.pop(session_id, None)

        now = time.time()
        idle_timeout_seconds = 30.0
        remaining_active: List[str] = []
        stale_to_drop: List[str] = []

        for sid, s in list(self.sessions.items()):
            if s.scene_id == scene_id:
                if (now - s.last_active) <= idle_timeout_seconds:
                    remaining_active.append(sid)
                else:
                    stale_to_drop.append(sid)

        for stale_sid in stale_to_drop:
            print(f"[Session Watchdog] Session [{stale_sid}] idle for >{idle_timeout_seconds:.1f}s. Evicting from active barrier...")
            self.sessions.pop(stale_sid, None)

        async with self.meta_lock:
            scene_meta = self.get_scene_metadata(scene_id)
            if session_id in scene_meta["active_sessions"]:
                scene_meta["active_sessions"].remove(session_id)
            for stale_sid in stale_to_drop:
                if stale_sid in scene_meta["active_sessions"]:
                    scene_meta["active_sessions"].remove(stale_sid)
            # Track the unique folder_name so multiple streams or scenes never collide
            if session.folder_name not in scene_meta["completed_sessions"]:
                scene_meta["completed_sessions"].append(session.folder_name)
            if session_id in scene_meta["completed_sessions"]:
                scene_meta["completed_sessions"].remove(session_id)
            scene_meta["active_sessions"] = sorted(remaining_active)
            self.save_scene_metadata(scene_id, scene_meta)
        # 4. Save Summary JSON (written before scene fusion so disk discovery finds it)
        elapsed = time.time() - session.created_at
        summary = {
            "type": "session_completed",
            "session_id": session_id,
            "scene_id": scene_id,
            "folder_name": session.folder_name,
            "created_at": session.created_at,
            "duration_seconds": round(elapsed, 2),
            "total_frames_received": session.frame_counter,
            "total_frames_processed": session.processed_counter,
            "raw_point_count": raw_count,
            "dedup_point_count": dedup_count,
            "voxel_size_m": session.voxel_size,
            "point_stride": session.point_stride,
            "frame_stride": session.frame_stride,
            "avg_fps": round(session.processed_counter / elapsed, 2) if elapsed > 0 else 0.0,
            "deliverables": deliverables,
            "scene_fusion": None,
            "scene_download_url": f"/api/scenes/{scene_id}/reconstruction.ply",
        }
        with open(session_dir / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 5. Barrier Coalescing (Option 1):
        # Only trigger global fusion when ALL streams in this scene have completed!
        scene_fusion_res: Optional[Dict[str, Any]] = None
        if len(remaining_active) > 0:
            print(
                f"\n[Scene Barrier] Session [{session_id}] finalized, but scene [{scene_id}] "
                f"still has {len(remaining_active)} active stream(s) in progress: {remaining_active}. "
                f"Deferring global fusion until all active streams complete."
            )
            scene_fusion_res = {
                "status": "deferred_barrier",
                "scene_id": scene_id,
                "message": f"Session completed. Waiting for {len(remaining_active)} other stream(s) to complete before full fusion: {remaining_active}",
                "waiting_for": remaining_active,
            }
        else:
            print(
                f"\n[Scene Barrier Cleared] All video streams for scene [{scene_id}] have completed! "
                f"Triggering unified full-scene fusion and optv2 optimization..."
            )
            if session.auto_fuse:
                try:
                    scene_fusion_res = await self.schedule_scene_fusion(
                        scene_id, voxel_size=session.voxel_size, debounce_seconds=1.0
                    )
                except Exception as e:
                    print(f"[Scene Auto-Fusion Notice] Auto-fusion for scene [{scene_id}] deferred/failed: {e}")
                    scene_fusion_res = {"status": "auto_fuse_failed", "error": str(e)}
        summary["scene_fusion"] = scene_fusion_res
        with open(session_dir / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Clean up session memory
        session.history_poses.clear()
        session.history_points.clear()
        session.history_colors.clear()
        session.history_conf.clear()
        # Notify real-time subscribers of stream conclusion
        self.broadcast_to_subscribers(session_id, {
            "type": "stream_end",
            "session_id": session_id,
            "scene_id": scene_id,
            "folder_name": session.folder_name,
            "total_frames": total_frames,
            "total_points": dedup_count if dedup_count > 0 else (total_frames * 5000),
            "ply_url": deliverables.get("ply_url", f"/api/streams/{session.folder_name}/reconstruction.ply"),
        })

        print(f"[Finalize] Session [{session_id}] finalized in {time.time() - t0:.2f}s! ({dedup_count:,} points) Scene [{scene_id}] updated.")
        return summary

    def _optimize_directory_sync(
        self,
        target_dir: Path,
        voxel_size: float = 0.015,
        mls_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Synchronous helper to run PostFusionOptimizer (optv2) on target_dir.
        Applies global PGO, ground plane leveling, bilateral surface thinning,
        submap-aware color correction, and dual-stage outlier removal (SOR+ROR).
        Updates standard reconstruction.ply, reconstruction_colored.ply, and transforms.json.
        """
        from scripts.optimize_fused_pointcloud import PostFusionOptimizer

        opt_dir = target_dir / "optimized"
        optimizer = PostFusionOptimizer(
            voxel_size=voxel_size,
            mls_iterations=mls_iterations,
            enable_submap_icp=True,
            enable_plane_leveling=True,
            enable_surface_thinning=True,
            enable_color_harmonization=True,
            enable_sor=True,
        )
        report = optimizer.optimize_fusion_directory(
            fusion_dir=target_dir,
            output_dir=opt_dir,
        )

        opt_normal = opt_dir / "fused_optimized_normal_merged.ply"
        opt_colored = opt_dir / "fused_optimized_colored_merged.ply"
        opt_transforms = opt_dir / "transforms.json"

        std_reconstruction = target_dir / "reconstruction.ply"
        std_colored = target_dir / "reconstruction_colored.ply"
        std_transforms = target_dir / "transforms.json"

        if opt_normal.is_file():
            shutil.copyfile(opt_normal, std_reconstruction)
        if opt_colored.is_file():
            shutil.copyfile(opt_colored, std_colored)
        if opt_transforms.is_file():
            shutil.copyfile(opt_transforms, std_transforms)

        return report

    def _fuse_incremental_to_scene_sync(
        self,
        scene_dir: Path,
        scene_id: str,
        existing_sids: List[str],
        new_sids: List[str],
        voxel_size: float = 0.015,
    ) -> Dict[str, Any]:
        """
        Incrementally registers new video streams directly against an already-optimized scene.
        1. Matches new_sid against existing sessions via Method 2 keyframe retrieval + LightGlue + Umeyama.
        2. Chains the new transform through the existing session's already-optimized transform in transforms.json.
        3. Refines new_sid's points directly against the optimized scene reconstruction.ply via small_gicp VGICP.
        4. Updates transforms.json with the new sessions.
        5. Runs PostFusionOptimizer (optv2) to thin seams and harmonize colors across the new combination.
        """
        transforms_file = scene_dir / "transforms.json"
        if not transforms_file.is_file():
            raise FileNotFoundError(f"Missing transforms.json in {scene_dir} for incremental fusion.")

        meta = json.loads(transforms_file.read_text(encoding="utf-8"))
        seq_dict = meta.get("sequences", {})
        anchor_id = meta.get("anchor_sequence", existing_sids[0])

        scene_ply = scene_dir / "reconstruction.ply"
        if not scene_ply.is_file():
            raise FileNotFoundError(f"Missing optimized reconstruction.ply in {scene_dir}")
        scene_pcd = o3d.io.read_point_cloud(str(scene_ply))
        scene_down = scene_pcd.voxel_down_sample(voxel_size)
        scene_down_pts = np.asarray(scene_down.points, dtype=np.float64)

        from scripts.match_and_fuse_method2 import Method2FusionPipeline, generate_distinct_palette, MatchEdge
        import small_gicp

        pipeline = Method2FusionPipeline(
            device=str(self.device),
            merge_voxel_size=voxel_size,
        )

        all_sids = existing_sids + new_sids
        recon_dirs = [self.output_base_dir / sid for sid in all_sids]
        items = pipeline.load_sequence_items(recon_dirs)

        palette_colors = generate_distinct_palette(len(all_sids))
        palette = {sid: palette_colors[i] for i, sid in enumerate(all_sids)}

        for new_sid in new_sids:
            item_new = items.get(new_sid)
            if item_new is None:
                raise ValueError(f"Could not load sequence items for {new_sid}")

            best_edge = None
            best_score = -1.0
            best_ex_sid = None

            for ex_sid in existing_sids:
                item_ex = items.get(ex_sid)
                if item_ex is None:
                    continue
                edge = pipeline.match_pair(item_new, item_ex)
                if edge is not None and edge.inliers >= 6 and edge.score > best_score:
                    best_score = edge.score
                    best_edge = edge
                    best_ex_sid = ex_sid

            if best_edge is None or best_ex_sid is None:
                for ex_sid in existing_sids:
                    item_ex = items.get(ex_sid)
                    if item_ex is None:
                        continue
                    edge_rev = pipeline.match_pair(item_ex, item_new)
                    if edge_rev is not None and edge_rev.inliers >= 6 and edge_rev.score > best_score:
                        inv_scale = 1.0 / edge_rev.scale
                        inv_T = np.linalg.inv(edge_rev.T_fine)
                        best_score = edge_rev.score
                        best_ex_sid = ex_sid
                        best_edge = MatchEdge(
                            src=new_sid,
                            tgt=ex_sid,
                            scale=inv_scale,
                            R=inv_T[:3, :3],
                            t=inv_T[:3, 3],
                            T_fine=inv_T,
                            inliers=edge_rev.inliers,
                            inlier_ratio=edge_rev.inlier_ratio,
                            rmse=edge_rev.rmse,
                            score=edge_rev.score,
                        )

            if best_edge is None or best_ex_sid is None:
                raise RuntimeError(
                    f"Incremental fusion failed: new session [{new_sid}] could not find overlapping "
                    f"keyframes with existing scene sessions {existing_sids}."
                )

            print(
                f"[Incremental Fusion] Matched new stream [{new_sid}] -> existing [{best_ex_sid}] "
                f"({best_edge.inliers} inliers, scale={best_edge.scale:.4f}, RMSE={best_edge.rmse*1000:.1f}mm)"
            )

            ex_info = seq_dict[best_ex_sid]
            ex_scale = float(ex_info.get("scale_to_anchor", 1.0))
            ex_T = np.array(ex_info.get("transform_matrix", np.eye(4)), dtype=np.float64)

            chained_scale = ex_scale * best_edge.scale
            T_scaled = best_edge.T_fine.copy()
            T_scaled[:3, 3] *= ex_scale
            chained_T = ex_T @ T_scaled

            # Fine registration directly against the OPTIMIZED scene point cloud using VGICP
            pts_init = (np.asarray(item_new.pcd_down.points, dtype=np.float64) * chained_scale) @ chained_T[:3, :3].T + chained_T[:3, 3]
            gicp_res = small_gicp.align(
                target_points=scene_down_pts,
                source_points=pts_init,
                init_T_target_source=np.eye(4, dtype=np.float64),
                registration_type="VGICP",
                voxel_resolution=0.08,
                downsampling_resolution=voxel_size,
                max_correspondence_distance=0.08,
                max_iterations=40,
                num_threads=8,
            )
            final_T = gicp_res.T_target_source @ chained_T
            final_scale = chained_scale

            seq_dict[new_sid] = {
                "raw_points": len(item_new.pcd_raw.points),
                "scale_to_anchor": round(float(final_scale), 6),
                "color_rgb": palette[new_sid],
                "transform_matrix": final_T.tolist(),
                "ply_path": str(self.output_base_dir / new_sid / "reconstruction.ply"),
                "dir_path": str(self.output_base_dir / new_sid),
            }
            existing_sids.append(new_sid)

        meta["sequences"] = seq_dict
        meta["total_raw_points"] = sum(s.get("raw_points", 0) for s in seq_dict.values())
        with open(transforms_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "status": "incremental_aligned",
            "new_sessions": new_sids,
            "all_sessions": all_sids,
            "anchor": anchor_id,
        }

    async def fuse_scene(
        self,
        scene_id: str,
        voxel_size: float = 0.015,
        anchor: Optional[str] = None,
        outputs: str = "normal,colored,transforms",
        force_full: bool = False,
    ) -> Dict[str, Any]:
        """Fuse all completed streaming sessions belonging to scene_id."""
        scene_dir = self.scenes_base_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        meta = self.get_scene_metadata(scene_id)
        completed = meta.get("completed_sessions", [])

        # Filter to sessions that have reconstruction.ply on disk
        valid_sids = [
            sid for sid in completed
            if (self.output_base_dir / sid / "reconstruction.ply").is_file()
        ]
        if not valid_sids:
            # Fallback check
            for d in self.output_base_dir.iterdir():
                if d.is_dir() and not d.is_symlink() and (d / "reconstruction.ply").is_file():
                    if d.name.startswith(f"{scene_id}-") or d.name.startswith(f"{scene_id}_") or d.name == scene_id:
                        valid_sids.append(d.name)
        if not valid_sids:
            raise ValueError(f"No completed sessions with point clouds found for scene '{scene_id}'")

        meta["completed_sessions"] = valid_sids

        # Case 1: Exactly 1 completed session for this scene -> Single stream baseline
        if len(valid_sids) == 1:
            sid = valid_sids[0]
            src_ply = self.output_base_dir / sid / "reconstruction.ply"
            dst_ply = scene_dir / "reconstruction.ply"
            shutil.copyfile(src_ply, dst_ply)

            src_traj = self.output_base_dir / sid / "camera_poses.npy"
            if src_traj.is_file():
                shutil.copyfile(src_traj, scene_dir / f"{sid}_poses.npy")
                robot_tag = sid.split("-")[-1] if "-" in sid else sid
                shutil.copyfile(src_traj, scene_dir / f"{robot_tag}_poses.npy")

            pcd = o3d.io.read_point_cloud(str(dst_ply))
            pt_count = len(pcd.points)
            size_mb = round(dst_ply.stat().st_size / 1024 / 1024, 2)

            meta["fusion_status"] = "single_stream"
            meta["is_fused"] = False
            meta["anchor_session"] = sid
            meta["total_points"] = pt_count
            meta["models"] = {
                "reconstruction_ply": f"/api/scenes/{scene_id}/reconstruction.ply",
                "ply_size_mb": size_mb,
            }
            self.save_scene_metadata(scene_id, meta)

            return {
                "status": "single_stream",
                "scene_id": scene_id,
                "message": f"Scene '{scene_id}' contains 1 video stream [{sid}]. Baseline reconstruction ready.",
                "total_points": pt_count,
                "ply_size_mb": size_mb,
                "download_url": f"/api/scenes/{scene_id}/reconstruction.ply",
            }

        # Case 2: 2 or more completed sessions for this scene -> Method 2 Multimodal Fusion
        # Check if this scene already has an optimized fusion and can be incrementally expanded
        std_reconstruction = scene_dir / "reconstruction.ply"
        std_transforms = scene_dir / "transforms.json"

        if not force_full and std_transforms.is_file() and std_reconstruction.is_file():
            try:
                prev_meta = json.loads(std_transforms.read_text(encoding="utf-8"))
                prev_sids = [s for s in prev_meta.get("sequences", {}).keys() if s in valid_sids]
                new_sids = [s for s in valid_sids if s not in prev_sids]

                if len(prev_sids) >= 1 and len(new_sids) > 0:
                    print(f"\n[Scene Incremental Fusion] Scene [{scene_id}] already has {len(prev_sids)} stream(s): {prev_sids}.")
                    print(f"  Fusing {len(new_sids)} new stream(s) directly against the optimized scene point cloud: {new_sids}...")
                    meta["fusion_status"] = "fusing_incremental"
                    self.save_scene_metadata(scene_id, meta)

                    async with self.gpu_lock:
                        loop = asyncio.get_event_loop()
                        inc_res = await loop.run_in_executor(
                            None,
                            lambda: self._fuse_incremental_to_scene_sync(
                                scene_dir=scene_dir,
                                scene_id=scene_id,
                                existing_sids=list(prev_sids),
                                new_sids=list(new_sids),
                                voxel_size=voxel_size,
                            ),
                        )

                    # Run optv2 optimization outside gpu_lock so GPU inference stays unblocked
                    print(f"[Incremental Fusion] Running optv2 post-fusion optimization on updated scene [{scene_id}]...")
                    try:
                        loop = asyncio.get_event_loop()
                        opt_report = await loop.run_in_executor(
                            None,
                            lambda: self._optimize_directory_sync(
                                target_dir=scene_dir,
                                voxel_size=voxel_size,
                                mls_iterations=3,
                            ),
                        )
                        inc_res["optimization_report"] = opt_report
                    except Exception as oe:
                        print(f"[Incremental Fusion Notice] optv2 optimization notice: {oe}")

                    pcd = o3d.io.read_point_cloud(str(std_reconstruction))
                    pt_count = len(pcd.points)
                    size_mb = round(std_reconstruction.stat().st_size / 1024 / 1024, 2)

                    meta["fusion_status"] = "fused"
                    meta["is_fused"] = True
                    meta["total_points"] = pt_count
                    meta["anchor_session"] = prev_meta.get("anchor_sequence", prev_sids[0])
                    meta["models"] = {
                        "reconstruction_ply": f"/api/scenes/{scene_id}/reconstruction.ply",
                        "reconstruction_colored_ply": f"/api/scenes/{scene_id}/reconstruction.ply?colored=true",
                        "transforms_json": f"/api/scenes/{scene_id}/transforms.json",
                        "ply_size_mb": size_mb,
                    }
                    meta["optimization_report"] = inc_res.get("optimization_report")
                    self.save_scene_metadata(scene_id, meta)

                    return {
                        "status": "fused",
                        "mode": "incremental_optimized",
                        "scene_id": scene_id,
                        "message": f"Successfully incrementally fused {len(new_sids)} new video stream(s) into optimized scene '{scene_id}'!",
                        "sessions": valid_sids,
                        "anchor_session": meta["anchor_session"],
                        "total_points": pt_count,
                        "ply_size_mb": size_mb,
                        "download_url": f"/api/scenes/{scene_id}/reconstruction.ply",
                        "download_colored_url": f"/api/scenes/{scene_id}/reconstruction.ply?colored=true",
                        "transforms_url": f"/api/scenes/{scene_id}/transforms.json",
                        "fusion_details": inc_res,
                    }
            except Exception as e:
                print(f"[Incremental Fusion Notice] Incremental fusion failed ({e}); falling back to full fusion...")

        # Case 2: Full Method 2 Multimodal Fusion across all streams
        print(f"\n[Scene Fusion] Fusing {len(valid_sids)} streams for scene [{scene_id}]: {valid_sids}...")
        meta["fusion_status"] = "fusing"
        self.save_scene_metadata(scene_id, meta)

        recon_dirs = [self.output_base_dir / sid for sid in valid_sids]

        # Verify required tensors
        missing_tensors = [
            sid for sid in valid_sids
            if not (self.output_base_dir / sid / "world_points.pt").is_file()
        ]
        if missing_tensors:
            raise ValueError(f"Sessions {missing_tensors} lack world_points.pt required for Method 2 fusion.")

        from scripts.match_and_fuse_method2 import Method2FusionPipeline

        async with self.gpu_lock:
            loop = asyncio.get_event_loop()
            def _run():
                pipeline = Method2FusionPipeline(
                    device=str(self.device),
                    merge_voxel_size=voxel_size,
                )
                return pipeline.execute(
                    inputs=recon_dirs,
                    output_dir=scene_dir,
                    outputs=outputs,
                    anchor=anchor,
                    prefix=scene_id,
                )
            result = await loop.run_in_executor(None, _run)

        normal_merged = scene_dir / f"{scene_id}_normal_merged.ply"
        colored_merged = scene_dir / f"{scene_id}_colored_merged.ply"
        transforms_file = scene_dir / f"{scene_id}_transforms.json"

        std_reconstruction = scene_dir / "reconstruction.ply"
        std_colored = scene_dir / "reconstruction_colored.ply"
        std_transforms = scene_dir / "transforms.json"

        if normal_merged.is_file():
            shutil.copyfile(normal_merged, std_reconstruction)
        if colored_merged.is_file():
            shutil.copyfile(colored_merged, std_colored)
        if transforms_file.is_file():
            shutil.copyfile(transforms_file, std_transforms)

        # Automatic Post-Fusion Optimization (optv2)
        print(f"\n[Scene Post-Fusion Optimization] Running optv2 on scene [{scene_id}]...")
        opt_report = None
        try:
            loop = asyncio.get_event_loop()
            opt_report = await loop.run_in_executor(
                None,
                lambda: self._optimize_directory_sync(
                    target_dir=scene_dir,
                    voxel_size=voxel_size,
                    mls_iterations=3,
                ),
            )
        except Exception as e:
            print(f"[Scene Optimization Notice] optv2 optimization failed ({e}), kept unoptimized fusion.")

        pcd = o3d.io.read_point_cloud(str(std_reconstruction))
        pt_count = len(pcd.points)
        size_mb = round(std_reconstruction.stat().st_size / 1024 / 1024, 2)

        meta["fusion_status"] = "fused"
        meta["is_fused"] = True
        meta["total_points"] = pt_count
        meta["anchor_session"] = result.get("anchor", valid_sids[0])
        meta["models"] = {
            "reconstruction_ply": f"/api/scenes/{scene_id}/reconstruction.ply",
            "reconstruction_colored_ply": f"/api/scenes/{scene_id}/reconstruction.ply?colored=true",
            "transforms_json": f"/api/scenes/{scene_id}/transforms.json",
            "ply_size_mb": size_mb,
        }
        if opt_report:
            meta["optimization_report"] = opt_report
        self.save_scene_metadata(scene_id, meta)

        return {
            "status": "fused",
            "scene_id": scene_id,
            "message": f"Successfully fused and optv2-optimized {len(valid_sids)} video streams for scene '{scene_id}'!",
            "sessions": valid_sids,
            "anchor_session": meta["anchor_session"],
            "total_points": pt_count,
            "ply_size_mb": size_mb,
            "download_url": f"/api/scenes/{scene_id}/reconstruction.ply",
            "download_colored_url": f"/api/scenes/{scene_id}/reconstruction.ply?colored=true",
            "transforms_url": f"/api/scenes/{scene_id}/transforms.json",
            "fusion_details": result,
            "optimization_report": opt_report,
        }
    async def fuse_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        outputs: str = "normal,colored,transforms,report,viewer",
        voxel_size: float = 0.015,
        anchor: Optional[str] = None,
        prefix: str = "stream_fused",
    ) -> Dict[str, Any]:
        """Trigger Method 2 multi-stream multimodal fusion across completed sessions."""
        from scripts.match_and_fuse_method2 import Method2FusionPipeline

        if not session_ids:
            session_ids = [
                d.name for d in self.output_base_dir.iterdir()
                if d.is_dir() and (d / "world_points.pt").is_file()
            ]

        if len(session_ids) < 2:
            raise ValueError(f"Method 2 fusion requires at least 2 sessions with point clouds, found: {session_ids}")

        recon_dirs = [self.output_base_dir / sid for sid in session_ids]
        fusion_out_dir = REPO_ROOT / "outputs/alignment/stream_fusions" / prefix
        fusion_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Server Fusion] Triggering Method 2 Fusion for sessions: {session_ids} -> {fusion_out_dir}")

        async with self.gpu_lock:
            loop = asyncio.get_event_loop()
            def _run():
                pipeline = Method2FusionPipeline(
                    device=str(self.device),
                    merge_voxel_size=voxel_size,
                )
                return pipeline.execute(
                    inputs=recon_dirs,
                    output_dir=fusion_out_dir,
                    outputs=outputs,
                    anchor=anchor,
                    prefix=prefix,
                )
            result = await loop.run_in_executor(None, _run)


        # Automatically run optv2 post-fusion optimization on stream fusions as well
        print(f"\n[Server Fusion Optimization] Running optv2 on {fusion_out_dir}...")
        try:
            loop = asyncio.get_event_loop()
            opt_report = await loop.run_in_executor(
                None,
                lambda: self._optimize_directory_sync(
                    target_dir=fusion_out_dir,
                    voxel_size=voxel_size,
                    mls_iterations=3,
                ),
            )
            result["optimization_report"] = opt_report
        except Exception as e:
            print(f"[Server Fusion Optimization Notice] optv2 optimization failed ({e}).")
        return result


# ---------------------------------------------------------------------------
# FastAPI Application & WebSocket Routes
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ABot-Recon Real-time Streaming 3D Reconstruction API",
    description="Multi-stream persistent WebSocket and REST API for real-time video point cloud construction.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MANAGER: Optional[MultiSessionReconstructionManager] = None


def get_manager() -> MultiSessionReconstructionManager:
    global _MANAGER
    if _MANAGER is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _MANAGER = MultiSessionReconstructionManager(device=device)
    return _MANAGER


@app.get("/")
def index():
    return {
        "service": "ABot-Recon Real-time Video Point Cloud Reconstruction Engine",
        "status": "online",
        "websocket_endpoint": "/ws/stream?session_id=<id>&scene_id=<scene_id>&point_stride=4&voxel_size=0.015",
        "scene_endpoints": {
            "list_scenes": "GET /api/scenes",
            "get_scene": "GET /api/scenes/{scene_id}",
            "download_scene_ply": "GET /api/scenes/{scene_id}/reconstruction.ply[?colored=true]",
            "trigger_scene_fusion": "POST /api/scenes/{scene_id}/fuse",
            "download_scene_transforms": "GET /api/scenes/{scene_id}/transforms.json",
        },
        "rest_endpoints": {
            "start_session": "POST /api/stream/start?session_id=<id>&scene_id=<scene_id>",
            "push_frame": "POST /api/stream/frame?session_id=<id>",
            "end_session": "POST /api/stream/end?session_id=<id>",
            "list_sessions": "GET /api/sessions",
            "download_session_ply": "GET /api/streams/{session_id}/reconstruction.ply",
            "multi_stream_fusion": "POST /api/fuse",
            "list_fusions": "GET /api/fusions",
        },
    }


# ---------------------------------------------------------------------------
# 1. WebSocket Streaming Endpoint (Bidirectional Real-time Persistent)
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str = Query(..., description="Unique persistent identifier for this video stream"),
    scene_id: Optional[str] = Query(None, description="Scene ID to group and fuse multiple video streams together"),
    robot: Optional[str] = Query(None, description="Robot identifier (e.g. Robot_A, Robot_B) for '场景-时间戳-机器人' directory naming"),
    point_stride: int = Query(4, description="Subsample stride for points (4=8.8k, 2=35k, 1=141k)"),
    frame_stride: int = Query(1, description="Process 1 frame every N frames"),
    voxel_size: float = Query(0.015, description="Final spatial voxel grid size in meters"),
    confidence_threshold: float = Query(0.1, description="Confidence threshold for points [0, 1]"),
    include_points: bool = Query(True, description="Whether to include point coordinates in frame response"),
    auto_fuse: bool = Query(True, description="Automatically fuse with existing completed streams in the same scene on EOS"),
    dynamic_filter: Optional[bool] = Query(None, description="Enable real-time dynamic object removal (YOLO-seg)"),
):
    await websocket.accept()
    manager = get_manager()
    session = manager.get_or_create_session(
        session_id=session_id,
        scene_id=scene_id,
        robot=robot,
        point_stride=point_stride,
        frame_stride=frame_stride,
        voxel_size=voxel_size,
        confidence_threshold=confidence_threshold,
        auto_fuse=auto_fuse,
        dynamic_filter=dynamic_filter,
    )

    # Initial handshake acknowledgement
    await websocket.send_text(
        json.dumps({
            "type": "session_connected",
            "session_id": session_id,
            "scene_id": session.scene_id,
            "robot": session.robot,
            "folder_name": session.folder_name,
            "status": "ready",
            "config": {
                "point_stride": session.point_stride,
                "frame_stride": session.frame_stride,
                "voxel_size": session.voxel_size,
                "confidence_threshold": session.confidence_threshold,
                "auto_fuse": session.auto_fuse,
                "dynamic_filter": session.dynamic_filter,
            },
        })
    )

    try:
        while True:
            message = await websocket.receive()

            # Case A: WebSocket Disconnect
            if message["type"] == "websocket.disconnect":
                break

            # Case B: Binary frame (fastest, raw JPEG/PNG image bytes)
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Check for binary EOS token
                if raw_bytes.startswith(b"EOS") or raw_bytes == b"__END_OF_STREAM__":
                    print(f"[WS] Received binary EOS token for session [{session_id}]")
                    summary = await manager.finalize_session(session_id)
                    await websocket.send_text(json.dumps(summary))
                    break

                res = await manager.process_frame(session, raw_bytes, include_point_data=include_points)
                await websocket.send_text(json.dumps(res))

            # Case C: Text / JSON message
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    payload = {"action": message["text"].strip()}

                # Check for End-of-Stream action
                msg_type = str(payload.get("type", payload.get("action", ""))).upper()
                if msg_type in ("EOS", "END", "FINISH", "STOP"):
                    print(f"[WS] Received text EOS command for session [{session_id}]")
                    summary = await manager.finalize_session(session_id)
                    if payload.get("fuse") or payload.get("fuse_with"):
                        fuse_sids = payload.get("fuse_with", [])
                        if session_id not in fuse_sids:
                            fuse_sids.append(session_id)
                        try:
                            fuse_res = await manager.fuse_sessions(
                                session_ids=fuse_sids,
                                outputs=payload.get("outputs", "normal,colored,transforms"),
                                voxel_size=float(payload.get("voxel_size", 0.015)),
                            )
                            summary["fusion_result"] = fuse_res
                        except Exception as fe:
                            summary["fusion_error"] = str(fe)
                    await websocket.send_text(json.dumps(summary))
                    break

                # Frame with base64 image
                if "image_b64" in payload:
                    import base64
                    img_bytes = base64.b64decode(payload["image_b64"])
                    res = await manager.process_frame(session, img_bytes, include_point_data=include_points)
                    await websocket.send_text(json.dumps(res))
                else:
                    await websocket.send_text(json.dumps({"type": "ack", "received": payload}))

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected gracefully for session [{session_id}]")
    except Exception as e:
        print(f"[WS Error] Session [{session_id}] error: {e}")
    finally:
        # Guarantee session finalization if not already closed
        if session_id in manager.sessions:
            await manager.finalize_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1.1. Real-time Viewer WebSocket and SSE Endpoints (Observer/Visualizer Bridge)
# ---------------------------------------------------------------------------
@app.websocket("/ws/viewer")
@app.websocket("/ws/viewer/{session_id}")
async def websocket_viewer(
    websocket: WebSocket,
    session_id: Optional[str] = None,
):
    """
    Real-time observer WebSocket for 3D visualizers (e.g. 8088 viewer).
    Broadcasts frame results (poses, incremental points, HUD images) as video is streamed.
    """
    await websocket.accept()
    manager = get_manager()
    target_session = session_id.strip() if (session_id and session_id.strip() not in ("*", "all")) else "*"
    q: asyncio.Queue = asyncio.Queue(maxsize=40)
    manager.add_subscriber(target_session, q)

    active_list = [
        {
            "session_id": sid,
            "scene_id": s.scene_id,
            "robot": s.robot,
            "folder_name": s.folder_name,
            "frames_received": s.frame_counter,
            "frames_processed": s.processed_counter,
        }
        for sid, s in manager.sessions.items() if s.is_active
    ]
    await websocket.send_text(json.dumps({
        "type": "viewer_connected",
        "target_session": target_session,
        "active_sessions": active_list,
    }))

    # If joining an ongoing session that has already reconstructed points, immediately catch up with history snapshot
    if target_session and target_session != "*":
        snapshot = manager.get_session_snapshot(target_session)
        if snapshot:
            await websocket.send_text(json.dumps(snapshot))

    try:
        while True:
            msg = await q.get()
            await websocket.send_text(json.dumps(msg))
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        manager.remove_subscriber(target_session, q)


@app.get("/api/stream/live")
@app.get("/api/stream/live/{session_id}")
async def sse_viewer(
    session_id: Optional[str] = None,
):
    """
    Server-Sent Events (SSE) live stream endpoint for web visualizers.
    Allows 8088 browser client to subscribe to any incoming video stream in real-time.
    """
    manager = get_manager()
    target_session = session_id.strip() if (session_id and session_id.strip() not in ("*", "all")) else "*"
    q: asyncio.Queue = asyncio.Queue(maxsize=40)
    manager.add_subscriber(target_session, q)

    async def event_generator():
        try:
            active_list = [
                {
                    "session_id": sid,
                    "scene_id": s.scene_id,
                    "robot": s.robot,
                    "folder_name": s.folder_name,
                    "frames_received": s.frame_counter,
                    "frames_processed": s.processed_counter,
                }
                for sid, s in manager.sessions.items() if s.is_active
            ]
            init_data = json.dumps({
                "type": "viewer_connected",
                "target_session": target_session,
                "active_sessions": active_list,
            })
            yield f"event: connect\ndata: {init_data}\n\n"

            # If joining an ongoing session that has already reconstructed points, immediately catch up with history snapshot
            if target_session and target_session != "*":
                snapshot = manager.get_session_snapshot(target_session)
                if snapshot:
                    yield f"event: history_sync\ndata: {json.dumps(snapshot)}\n\n"

            while True:
                msg = await q.get()
                ev_type = msg.get("type", "frame_result")
                yield f"event: {ev_type}\ndata: {json.dumps(msg)}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            manager.remove_subscriber(target_session, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/stream/snapshot/{session_id}")
async def get_stream_snapshot(session_id: str):
    """Fetch current accumulated 3D reconstruction snapshot of an active stream."""
    manager = get_manager()
    snapshot = manager.get_session_snapshot(session_id.strip())
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"No active point snapshot for session '{session_id}'")
    return snapshot


# ---------------------------------------------------------------------------
# 2. REST Endpoints (Alternative for Non-WebSocket Clients)
# ---------------------------------------------------------------------------
@app.post("/api/stream/start")
def start_session(
    session_id: str = Query(...),
    scene_id: Optional[str] = Query(None, description="Scene ID to associate this stream with"),
    robot: Optional[str] = Query(None, description="Robot ID/Name, e.g. Robot_A, Robot_B"),
    point_stride: int = Query(4),
    frame_stride: int = Query(1),
    voxel_size: float = Query(0.015),
    confidence_threshold: float = Query(0.1),
    auto_fuse: bool = Query(True, description="Whether to auto-fuse when this stream ends"),
    dynamic_filter: Optional[bool] = Query(None, description="Enable real-time dynamic object removal (YOLO-seg)"),
):
    """Initialize a new persistent streaming session."""
    manager = get_manager()
    session = manager.get_or_create_session(
        session_id=session_id,
        scene_id=scene_id,
        robot=robot,
        point_stride=point_stride,
        frame_stride=frame_stride,
        voxel_size=voxel_size,
        confidence_threshold=confidence_threshold,
        auto_fuse=auto_fuse,
        dynamic_filter=dynamic_filter,
    )
    return {
        "status": "session_started",
        "session_id": session.session_id,
        "scene_id": session.scene_id,
        "robot": session.robot,
        "folder_name": session.folder_name,
        "scene_download_url": f"/api/scenes/{session.scene_id}/reconstruction.ply",
        "stream_download_url": f"/api/streams/{session.folder_name}/reconstruction.ply",
        "config": {
            "point_stride": session.point_stride,
            "frame_stride": session.frame_stride,
            "voxel_size": session.voxel_size,
            "confidence_threshold": session.confidence_threshold,
            "auto_fuse": session.auto_fuse,
            "dynamic_filter": session.dynamic_filter,
        },
    }


@app.post("/api/stream/frame")
async def push_frame(
    request: Request,
    session_id: str = Query(...),
    include_points: bool = Query(False),
):
    """Post a single frame as raw binary image body to an active session."""
    manager = get_manager()
    if session_id not in manager.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /api/stream/start first.")

    session = manager.sessions[session_id]
    image_bytes = await request.body()
    result = await manager.process_frame(session, image_bytes, include_point_data=include_points)
    return result


@app.post("/api/stream/end")
async def end_session(session_id: str = Query(...)):
    """Explicitly signal End-of-Stream (EOS) to finalize and export the point cloud."""
    manager = get_manager()
    if session_id not in manager.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not active.")
    summary = await manager.finalize_session(session_id)
    return summary


@app.get("/api/sessions")
def list_sessions():
    """List all currently active sessions and completed session directories."""
    manager = get_manager()
    now = time.time()
    active = [
        {
            "session_id": sid,
            "scene_id": s.scene_id,
            "robot": s.robot,
            "folder_name": s.folder_name,
            "frames_received": s.frame_counter,
            "frames_processed": s.processed_counter,
            "uptime_seconds": round(now - s.created_at, 1),
            "last_active_seconds_ago": round(now - s.last_active, 1),
            "is_active": s.is_active,
        }
        for sid, s in manager.sessions.items()
        if s.is_active
    ]
    completed = []
    if manager.output_base_dir.is_dir():
        for d in sorted(manager.output_base_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if d.is_dir() and not d.is_symlink() and (d / "reconstruction.ply").is_file():
                ply = d / "reconstruction.ply"
                completed.append({
                    "session_id": d.name,
                    "folder_name": d.name,
                    "ply_size_mb": round(ply.stat().st_size / 1024 / 1024, 2),
                    "download_url": f"/api/streams/{d.name}/reconstruction.ply",
                })
    return {"active_sessions": active, "completed_sessions": completed}


@app.get("/api/streams/{session_id}/reconstruction.ply")
def download_session_ply(session_id: str):
    """Download the generated PLY file for a completed session (supports session_id or folder_name)."""
    manager = get_manager()
    target_dir = manager.output_base_dir / session_id
    if not target_dir.is_dir():
        # Match by folder_name or suffix
        for d in manager.output_base_dir.iterdir():
            if d.is_dir() and (d.name == session_id or d.name.endswith(f"-{session_id}")):
                target_dir = d
                break
    ply = target_dir / "reconstruction.ply"
    if not ply.is_file():
        raise HTTPException(status_code=404, detail=f"PLY not found for session '{session_id}'")
    return FileResponse(str(ply), media_type="application/octet-stream", filename=f"{target_dir.name}_reconstruction.ply")


@app.post("/api/fuse")
async def fuse_streams(
    session_ids: Optional[str] = Query(None, description="Comma-separated session IDs to fuse, e.g. 'cam_1,cam_2'. Omit to fuse all."),
    outputs: str = Query("normal,colored,transforms,report,viewer", description="Outputs: normal, colored, all, etc."),
    voxel_size: float = Query(0.015, description="Voxel size in meters"),
    anchor: Optional[str] = Query(None, description="Anchor session ID (default: auto)"),
    prefix: str = Query("stream_fused", description="Output filename prefix"),
):
    """Trigger Method 2 multimodal registration & fusion across completed video streams."""
    manager = get_manager()
    sids = [s.strip() for s in session_ids.split(",") if s.strip()] if session_ids else None
    try:
        res = await manager.fuse_sessions(
            session_ids=sids,
            outputs=outputs,
            voxel_size=voxel_size,
            anchor=anchor,
            prefix=prefix,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/fusions")
def list_fusions():
    """List all Method 2 fused models generated from streaming sessions."""
    fusions_dir = REPO_ROOT / "outputs/alignment/stream_fusions"
    items = []
    if fusions_dir.is_dir():
        for d in fusions_dir.iterdir():
            if d.is_dir():
                ply_files = list(d.glob("*.ply"))
                json_files = list(d.glob("*_transforms.json"))
                items.append({
                    "fusion_id": d.name,
                    "path": str(d),
                    "ply_models": [
                        {
                            "name": p.name,
                            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                            "download_url": f"/api/fusions/{d.name}/{p.name}",
                        }
                        for p in ply_files
                    ],
                    "metadata": str(json_files[0]) if json_files else None,
                })
    return {"fusions": items}


@app.get("/api/fusions/{fusion_id}/{filename}")
def download_fusion_file(fusion_id: str, filename: str):
    """Download a fused PLY model or transform JSON."""
    fpath = REPO_ROOT / "outputs/alignment/stream_fusions" / fusion_id / filename
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="Fusion deliverable not found")
    return FileResponse(str(fpath), media_type="application/octet-stream", filename=filename)



# ---------------------------------------------------------------------------
# 3. Scene Management & Fused Multi-Stream Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/scenes")
def list_scenes():
    """List all registered scenes, their component sessions, and fusion deliverables."""
    manager = get_manager()
    scenes_dir = manager.scenes_base_dir
    results = []
    if scenes_dir.is_dir():
        for d in sorted(scenes_dir.iterdir()):
            if d.is_dir():
                meta = manager.get_scene_metadata(d.name)
                ply = d / "reconstruction.ply"
                colored_ply = d / "reconstruction_colored.ply"
                results.append({
                    "scene_id": d.name,
                    "fusion_status": meta.get("fusion_status", "none"),
                    "is_fused": meta.get("is_fused", False),
                    "completed_sessions": meta.get("completed_sessions", []),
                    "active_sessions": [
                        sid for sid, s in manager.sessions.items()
                        if getattr(s, "scene_id", "") == d.name
                    ],
                    "total_points": meta.get("total_points", 0),
                    "ply_exists": ply.is_file(),
                    "ply_size_mb": round(ply.stat().st_size / 1024 / 1024, 2) if ply.is_file() else None,
                    "colored_ply_exists": colored_ply.is_file(),
                    "download_url": f"/api/scenes/{d.name}/reconstruction.ply" if ply.is_file() else None,
                    "download_colored_url": f"/api/scenes/{d.name}/reconstruction.ply?colored=true" if colored_ply.is_file() else None,
                    "transforms_url": f"/api/scenes/{d.name}/transforms.json" if (d / "transforms.json").is_file() else None,
                    "updated_at": meta.get("updated_at"),
                })
    return {"scenes": results}


@app.get("/api/scenes/{scene_id}")
def get_scene_info(scene_id: str):
    """Get detailed status, component streams, and download links for a specific scene."""
    manager = get_manager()
    meta = manager.get_scene_metadata(scene_id)
    scene_dir = manager.scenes_base_dir / scene_id
    if not scene_dir.is_dir() and not meta.get("completed_sessions") and not meta.get("active_sessions"):
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")

    ply = scene_dir / "reconstruction.ply"
    colored_ply = scene_dir / "reconstruction_colored.ply"
    transforms = scene_dir / "transforms.json"

    meta["active_sessions"] = [
        sid for sid, s in manager.sessions.items()
        if getattr(s, "scene_id", "") == scene_id
    ]
    meta["deliverables"] = {
        "reconstruction_ply": f"/api/scenes/{scene_id}/reconstruction.ply" if ply.is_file() else None,
        "colored_ply": f"/api/scenes/{scene_id}/reconstruction.ply?colored=true" if colored_ply.is_file() else None,
        "transforms_json": f"/api/scenes/{scene_id}/transforms.json" if transforms.is_file() else None,
    }
    return meta


@app.get("/api/scenes/{scene_id}/reconstruction.ply")
async def download_scene_ply(
    scene_id: str,
    colored: bool = Query(False, description="Whether to download distinctly color-coded point cloud"),
):
    """Download the unified 3D point cloud model for a scene (auto-fuses multi-stream if needed)."""
    manager = get_manager()
    scene_dir = manager.scenes_base_dir / scene_id
    target_name = "reconstruction_colored.ply" if colored else "reconstruction.ply"
    ply_path = scene_dir / target_name

    # If file not yet on disk, attempt generation if completed sessions exist
    if not ply_path.is_file():
        meta = manager.get_scene_metadata(scene_id)
        if meta.get("completed_sessions"):
            try:
                await manager.fuse_scene(scene_id)
            except Exception as e:
                print(f"[Download Fusion Error] Failed auto-generating scene [{scene_id}]: {e}")

    # Fallback to standard reconstruction.ply if colored was requested but not generated
    if not ply_path.is_file() and colored:
        ply_path = scene_dir / "reconstruction.ply"

    if not ply_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Reconstruction model not found for scene '{scene_id}'. Push video streams to this scene first.",
        )

    out_name = f"{scene_id}_colored.ply" if (colored and "colored" in ply_path.name) else f"{scene_id}_reconstruction.ply"
    return FileResponse(str(ply_path), media_type="application/octet-stream", filename=out_name)


@app.post("/api/scenes/{scene_id}/fuse")
async def trigger_scene_fusion(
    scene_id: str,
    voxel_size: float = Query(0.015, description="Spatial voxel size in meters for de-duplication"),
    anchor: Optional[str] = Query(None, description="Anchor stream session ID (default: auto)"),
    force_full: bool = Query(True, description="Force full joint fusion across all streams instead of incremental"),
):
    """Explicitly trigger or re-run multi-stream multimodal fusion across all streams in this scene."""
    manager = get_manager()
    try:
        res = await manager.fuse_scene(scene_id=scene_id, voxel_size=voxel_size, anchor=anchor, force_full=force_full)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/scenes/{scene_id}/transforms.json")
def download_scene_transforms(scene_id: str):
    """Download the coordinate alignment transformations and scale factors for each stream in this scene."""
    manager = get_manager()
    scene_dir = manager.scenes_base_dir / scene_id
    tf_path = scene_dir / "transforms.json"
    if not tf_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transforms JSON not found for scene '{scene_id}'")
    return FileResponse(str(tf_path), media_type="application/json", filename=f"{scene_id}_transforms.json")
def main() -> None:
    parser = argparse.ArgumentParser(description="ABot-Recon Real-time Video Stream API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dynamic-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable real-time 2D dynamic object removal (YOLO-seg) by default",
    )
    parser.add_argument("--dynamic-model", type=str, default="yolo11m-seg.pt", help="YOLO segmentation model")
    parser.add_argument("--dynamic-conf", type=float, default=0.12, help="Confidence threshold for dynamic object detection")
    parser.add_argument("--dynamic-dilate", type=int, default=15, help="Dilation kernel size in pixels")
    args = parser.parse_args()

    # Pre-init manager
    global _MANAGER
    _MANAGER = MultiSessionReconstructionManager(
        device=args.device,
        dynamic_filter=args.dynamic_filter,
        dynamic_model=args.dynamic_model,
        dynamic_conf=args.dynamic_conf,
        dynamic_dilate=args.dynamic_dilate,
    )
    print(f"\n==================================================================")
    print(f"  ABot-Recon Streaming Server Running on http://{args.host}:{args.port}")
    print(f"  WebSocket Stream Endpoint: ws://{args.host}:{args.port}/ws/stream")
    print(f"==================================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
