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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image


# ---------------------------------------------------------------------------
# Session Data Structure
# ---------------------------------------------------------------------------
@dataclass
class StreamSession:
    session_id: str
    point_stride: int = 4
    frame_stride: int = 1
    voxel_size: float = 0.015
    confidence_threshold: float = 0.1
    save_ply: bool = True
    save_trajectory: bool = True

    # State tracking
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
    ):
        self.device = torch.device(device)
        self.output_base_dir = output_base_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, StreamSession] = {}
        self.gpu_lock = asyncio.Lock()

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

    def get_or_create_session(
        self,
        session_id: str,
        point_stride: int = 4,
        frame_stride: int = 1,
        voxel_size: float = 0.015,
        confidence_threshold: float = 0.1,
        save_ply: bool = True,
        save_trajectory: bool = True,
    ) -> StreamSession:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.point_stride = point_stride
            session.frame_stride = frame_stride
            session.voxel_size = voxel_size
            session.confidence_threshold = confidence_threshold
            session.is_active = True
            session.last_active = time.time()
            return session

        # Create fresh isolated paged manager
        paged_mgr = copy.deepcopy(self.paged_template)
        paged_mgr.reset()

        session = StreamSession(
            session_id=session_id,
            point_stride=max(1, int(point_stride)),
            frame_stride=max(1, int(frame_stride)),
            voxel_size=max(0.0, float(voxel_size)),
            confidence_threshold=max(0.0, min(1.0, float(confidence_threshold))),
            save_ply=save_ply,
            save_trajectory=save_trajectory,
            paged_manager=paged_mgr,
        )
        self.sessions[session_id] = session
        print(f"[Session] Created new session [{session_id}] (total active: {len(self.sessions)})")
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

        stride = session.point_stride
        thresh = session.confidence_threshold

        sampled_pts = raw_world[::stride, ::stride].reshape(-1, 3).numpy()
        sampled_conf = conf_map[::stride, ::stride].reshape(-1).numpy()
        valid = np.isfinite(sampled_pts).all(axis=-1)
        if thresh > 0:
            valid &= sampled_conf >= thresh

        valid_pts = sampled_pts[valid].astype(np.float32)
        rgb_arr = (tensor_chw.permute(1, 2, 0)[::stride, ::stride].reshape(-1, 3).numpy() * 255).astype(np.uint8)
        valid_rgb = rgb_arr[valid]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_pts_approx = session.processed_counter * len(valid_pts)

        result: Dict[str, Any] = {
            "type": "frame_result",
            "session_id": session.session_id,
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
        return result

    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize a session, export PLY model, save trajectory, and clean up resources."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' does not exist")

        session = self.sessions[session_id]
        session.is_active = False
        t0 = time.time()
        session_dir = self.output_base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

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

            stride = session.point_stride
            thresh = session.confidence_threshold

            flat_pts = all_pts[:, ::stride, ::stride].reshape(-1, 3).numpy()
            flat_rgb = all_colors[:, ::stride, ::stride].reshape(-1, 3).numpy()
            flat_conf = all_conf[:, ::stride, ::stride].reshape(-1).numpy()

            valid = np.isfinite(flat_pts).all(axis=-1)
            if thresh > 0:
                valid &= flat_conf >= thresh

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
            o3d.io.write_point_cloud(str(ply_path), final_pcd)
            deliverables["ply_path"] = str(ply_path)
            deliverables["ply_url"] = f"/api/streams/{session_id}/reconstruction.ply"
            deliverables["ply_size_mb"] = round(ply_path.stat().st_size / 1024 / 1024, 2)

        # 2. Save Trajectory
        if session.save_trajectory and len(session.history_poses) > 0:
            poses_np = np.stack(session.history_poses)
            pose_path = session_dir / "camera_poses.npy"
            np.save(pose_path, poses_np)
            deliverables["trajectory_path"] = str(pose_path)

        # 3. Save Summary JSON
        elapsed = time.time() - session.created_at
        summary = {
            "type": "session_completed",
            "session_id": session_id,
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
        }
        with open(session_dir / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Clean up session memory
        session.history_poses.clear()
        session.history_points.clear()
        session.history_colors.clear()
        session.history_conf.clear()
        session.paged_manager = None
        self.sessions.pop(session_id, None)

        print(f"[Finalize] Session [{session_id}] finalized in {time.time() - t0:.2f}s! ({dedup_count:,} points)")
        return summary
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
        "websocket_endpoint": "/ws/stream?session_id=<id>&point_stride=4&voxel_size=0.015",
        "rest_endpoints": {
            "start_session": "POST /api/stream/start",
            "push_frame": "POST /api/stream/frame",
            "end_session": "POST /api/stream/end",
            "list_sessions": "GET /api/sessions",
        },
    }


# ---------------------------------------------------------------------------
# 1. WebSocket Streaming Endpoint (Bidirectional Real-time Persistent)
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str = Query(..., description="Unique persistent identifier for this video stream"),
    point_stride: int = Query(4, description="Subsample stride for points (4=8.8k, 2=35k, 1=141k)"),
    frame_stride: int = Query(1, description="Process 1 frame every N frames"),
    voxel_size: float = Query(0.015, description="Final spatial voxel grid size in meters"),
    confidence_threshold: float = Query(0.1, description="Confidence threshold for points [0, 1]"),
    include_points: bool = Query(True, description="Whether to include point coordinates in frame response"),
):
    await websocket.accept()
    manager = get_manager()
    session = manager.get_or_create_session(
        session_id=session_id,
        point_stride=point_stride,
        frame_stride=frame_stride,
        voxel_size=voxel_size,
        confidence_threshold=confidence_threshold,
    )

    # Initial handshake acknowledgement
    await websocket.send_text(
        json.dumps({
            "type": "session_connected",
            "session_id": session_id,
            "status": "ready",
            "config": {
                "point_stride": session.point_stride,
                "frame_stride": session.frame_stride,
                "voxel_size": session.voxel_size,
                "confidence_threshold": session.confidence_threshold,
            },
            "end_stream_instruction": "Send text '{\"type\": \"EOS\"}' or binary token b'EOS\\x00\\x00' or close socket",
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
# 2. REST Endpoints (Alternative for Non-WebSocket Clients)
# ---------------------------------------------------------------------------
@app.post("/api/stream/start")
def start_session(
    session_id: str = Query(...),
    point_stride: int = Query(4),
    frame_stride: int = Query(1),
    voxel_size: float = Query(0.015),
    confidence_threshold: float = Query(0.1),
):
    """Initialize a new persistent streaming session."""
    manager = get_manager()
    session = manager.get_or_create_session(
        session_id=session_id,
        point_stride=point_stride,
        frame_stride=frame_stride,
        voxel_size=voxel_size,
        confidence_threshold=confidence_threshold,
    )
    return {
        "status": "session_started",
        "session_id": session.session_id,
        "config": {
            "point_stride": session.point_stride,
            "frame_stride": session.frame_stride,
            "voxel_size": session.voxel_size,
            "confidence_threshold": session.confidence_threshold,
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
    active = [
        {
            "session_id": sid,
            "frames_received": s.frame_counter,
            "frames_processed": s.processed_counter,
            "uptime_seconds": round(time.time() - s.created_at, 2),
        }
        for sid, s in manager.sessions.items()
    ]
    completed = []
    if manager.output_base_dir.is_dir():
        for d in manager.output_base_dir.iterdir():
            if d.is_dir() and (d / "reconstruction.ply").is_file():
                ply = d / "reconstruction.ply"
                completed.append({
                    "session_id": d.name,
                    "ply_size_mb": round(ply.stat().st_size / 1024 / 1024, 2),
                    "download_url": f"/api/streams/{d.name}/reconstruction.ply",
                })
    return {"active_sessions": active, "completed_sessions": completed}


@app.get("/api/streams/{session_id}/reconstruction.ply")
def download_session_ply(session_id: str):
    """Download the generated PLY file for a completed session."""
    manager = get_manager()
    ply = manager.output_base_dir / session_id / "reconstruction.ply"
    if not ply.is_file():
        raise HTTPException(status_code=404, detail=f"PLY not found for session '{session_id}'")
    return FileResponse(str(ply), media_type="application/octet-stream", filename=f"{session_id}_reconstruction.ply")



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

def main() -> None:
    parser = argparse.ArgumentParser(description="ABot-Recon Real-time Video Stream API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Pre-init manager
    global _MANAGER
    _MANAGER = MultiSessionReconstructionManager(device=args.device)

    print(f"\n==================================================================")
    print(f"  ABot-Recon Streaming Server Running on http://{args.host}:{args.port}")
    print(f"  WebSocket Stream Endpoint: ws://{args.host}:{args.port}/ws/stream")
    print(f"==================================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
