#!/usr/bin/env python3
"""Multi-threaded HTTP & Server-Sent Events (SSE) Streaming Server for ABot-Recon 3D Visualizer."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import base64
import cv2
import numpy as np
import open3d as o3d
import torch
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from viewer.stream_backend import OnlineReconstructionEngine

# In-memory cache for colors.pt video stream tensors (max 8 models)
_COLORS_CACHE: Dict[str, torch.Tensor] = {}

def load_colors_tensor(dir_or_file: Path) -> Optional[torch.Tensor]:
    """Load colors.pt from directory or parent directory with caching."""
    if dir_or_file.is_file():
        colors_file = dir_or_file.parent / "colors.pt"
    else:
        colors_file = dir_or_file / "colors.pt"

    if not colors_file.is_file():
        return None

    key = str(colors_file.resolve())
    if key in _COLORS_CACHE:
        return _COLORS_CACHE[key]

    try:
        tensor = torch.load(colors_file, map_location="cpu", weights_only=False)
        if isinstance(tensor, torch.Tensor):
            if len(_COLORS_CACHE) > 8:
                _COLORS_CACHE.pop(next(iter(_COLORS_CACHE)))
            _COLORS_CACHE[key] = tensor
            return tensor
    except Exception as e:
        print(f"[Colors] Failed to load {colors_file}: {e}", flush=True)
    return None

def encode_frame_jpeg_base64(colors_tensor: torch.Tensor, frame_idx: int, quality: int = 80) -> Optional[str]:
    """Extract frame_idx from colors_tensor and return data:image/jpeg;base64,... string."""
    if frame_idx < 0 or frame_idx >= len(colors_tensor):
        return None
    try:
        img_rgb = colors_tensor[frame_idx].numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return None
        b64 = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None



def get_ply_header_info(path: Path) -> Tuple[Optional[int], float]:
    """Quickly read PLY header to extract vertex count and file size in MB."""
    vertex_count = None
    try:
        with open(path, "rb") as f:
            head = f.read(1024).decode("latin1", errors="ignore")
            for line in head.split("\n"):
                line = line.strip()
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) >= 3:
                        vertex_count = int(parts[2])
                        break
    except Exception:
        pass
    size_mb = path.stat().st_size / (1024 * 1024)
    return vertex_count, size_mb

def format_point_count(count: Optional[int]) -> str:
    if count is None:
        return ""
    if count >= 10000:
        return f"{count / 10000:.1f}万点"
    return f"{count:,}点"

_SCAN_CACHE: Tuple[float, List[Dict[str, Any]]] = (0.0, [])

def find_trajectory_for_ply(ply_path: Path) -> Optional[str]:
    """Find matching .npy camera poses / trajectory file for a PLY model, if one exists."""
    ply_path = ply_path.resolve()
    parent = ply_path.parent
    stem = ply_path.stem

    candidates = [
        parent / f"{stem}_poses.npy",
        parent / f"{stem}_camera_poses.npy",
        parent / f"{stem}.poses.npy",
    ]
    if "reconstruction" in stem:
        candidates.extend([
            parent / "camera_poses.npy",
            parent / "camera_poses_loop.npy",
            parent / "camera_poses_noloop.npy",
        ])

    for c in candidates:
        if c.is_file():
            try:
                rel = c.relative_to(ROOT_DIR)
                return f"/{rel}"
            except ValueError:
                pass
    return None

def scan_all_ply_models(force: bool = False) -> List[Dict[str, Any]]:
    """Scan outputs/ directory dynamically with 3-second cache to prevent redundant disk I/O."""
    global _SCAN_CACHE
    now = time.time()
    if not force and (now - _SCAN_CACHE[0] < 3.0):
        return _SCAN_CACHE[1]

    outputs_dir = ROOT_DIR / "outputs"
    if not outputs_dir.is_dir():
        return []

    ply_files = sorted(outputs_dir.glob("**/*.ply"))
    models = []
    category_priority = {
        "🔥 多视角融合点云 (Multi-Stream Fusion)": 0,
        "📹 单视频重建点云 (Single Video)": 1,
        "🧹 滤波去噪点云 (Denoised Cleaned)": 2,
        "🛡️ 动态过滤静态底图 (Filtered Static Map)": 3,
        "🚶 原始含动态基线 (Baseline with Dynamic)": 4,
        "📦 仅动态物体 (Removed Dynamic Only)": 5,
        "🏛️ 场景全景融合 (Scene Fusion)": 6,
        "📡 实时流式会话 (Streaming Sessions)": 7,
        "🔄 回环与轨迹对比 (Loop Comparison)": 8,
        "📁 输出点云 (Outputs Point Clouds)": 9,
    }

    for ply_path in ply_files:
        rel_str = str(ply_path.relative_to(ROOT_DIR))
        url = f"/{rel_str}"
        vertex_count, size_mb = get_ply_header_info(ply_path)
        size_bytes = ply_path.stat().st_size
        mtime = ply_path.stat().st_mtime
        mtime_str = time.strftime("%m-%d %H:%M", time.localtime(mtime))
        folder_name = ply_path.parent.name
        stem = ply_path.stem
        fname = ply_path.name

        # 1. Multi-Stream / General Fusion
        if "fusion" in rel_str or "merged" in stem or "fused_" in stem:
            category = "🔥 多视角融合点云 (Multi-Stream Fusion)"
            is_colored = "colored" in stem
            is_full = "full" in stem
            tag = "🎨 [区分色彩]" if is_colored else "🌟 [正常真彩]"
            mode_str = "全量无损" if is_full else "体素去重"
            if "fused_" in stem:
                parts = stem.split("_")
                num_streams = f"{parts[1]}路" if len(parts) > 1 and parts[1].isdigit() else ""
                name = f"{tag} {num_streams}融合-{mode_str} ({folder_name}/{stem}, {format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
            else:
                name = f"{tag} {stem} ({mode_str}, {format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 2. Dynamic Filtering Contrast
        elif fname in ("reconstruction_clean.ply", "reconstruction_baseline_with_dynamic.ply", "reconstruction_removed_dynamic_only.ply", "reconstruction_scheme1_filtered.ply") or "dynamic" in fname or "clean" in fname:
            if "clean" in fname:
                category = "🧹 滤波去噪点云 (Denoised Cleaned)"
                tag = "🧹 [滤波去噪]"
            elif "baseline" in fname:
                category = "🚶 原始含动态基线 (Baseline with Dynamic)"
                tag = "🚶 [含动态基线]"
            elif "removed" in fname or "dynamic_only" in fname:
                category = "📦 仅动态物体 (Removed Dynamic Only)"
                tag = "📦 [仅动态物体]"
            else:
                category = "🛡️ 动态过滤静态底图 (Filtered Static Map)"
                tag = "🛡️ [静态过滤]"
            name = f"{tag} {folder_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 3. Single Video Reconstructions
        elif fname == "reconstruction.ply" or (ply_path.parent / "metadata.json").is_file():
            category = "📹 单视频重建点云 (Single Video)"
            name = f"📹 [单视频重建] {folder_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 4. Multi-Stream Scenes
        elif "scenes" in rel_str:
            category = "🏛️ 场景全景融合 (Scene Fusion)"
            is_colored = "colored" in stem
            tag = "🎨 [场景多色]" if is_colored else "🏛️ [场景全景]"
            name = f"{tag} {folder_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 5. Streaming Sessions
        elif "streams" in rel_str:
            category = "📡 实时流式会话 (Streaming Sessions)"
            name = f"📡 [流式会话] {folder_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 6. Loop Comparison
        elif "loop" in rel_str:
            category = "🔄 回环与轨迹对比 (Loop Comparison)"
            tag = "🔄 [回环优化]" if "loop" in stem else "🚀 [原始流式]"
            name = f"{tag} {folder_name}/{stem} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 7. Other Arbitrary Outputs
        else:
            category = "📁 输出点云 (Outputs Point Clouds)"
            name = f"📁 {folder_name}/{fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"

        description = f"路径: {rel_str} | 大小: {size_mb:.2f}MB | 点数: {vertex_count:,} | 时间: {mtime_str}" if vertex_count else f"路径: {rel_str} | 大小: {size_mb:.2f}MB | 时间: {mtime_str}"

        models.append({
            "url": url,
            "path": rel_str,
            "name": name,
            "category": category,
            "description": description,
            "size_bytes": size_bytes,
            "vertex_count": vertex_count,
            "mtime": mtime,
            "trajectory_url": find_trajectory_for_ply(ply_path),
            "_prio": category_priority.get(category, 50),
        })

    # Sort models: Primary key is category priority, Secondary key is mtime descending (newest files on top!)
    models.sort(key=lambda m: (m["_prio"], -m["mtime"], m["name"]))
    for m in models:
        m.pop("_prio", None)
    _SCAN_CACHE = (now, models)
    return models

def generate_orbital_poses(points_xyz: np.ndarray, num_poses: int = 60) -> List[List[List[float]]]:
    """Generate smooth circular camera trajectory orbiting the point cloud."""
    min_pt = points_xyz.min(axis=0)
    max_pt = points_xyz.max(axis=0)
    center = (min_pt + max_pt) / 2.0
    extent = float(np.linalg.norm(max_pt - min_pt))
    radius = max(extent * 0.8, 1.0)
    poses = []
    for i in range(num_poses):
        theta = 2.0 * np.pi * i / num_poses
        cam_pos = center + np.array([radius * np.cos(theta), -radius * 0.25, radius * np.sin(theta)], dtype=np.float32)
        forward = center - cam_pos
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_ref = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_ref)
        r_norm = float(np.linalg.norm(right))
        right = right / r_norm if r_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.cross(right, forward)

        mat = np.eye(4, dtype=np.float32)
        mat[0:3, 0] = right
        mat[0:3, 1] = up
        mat[0:3, 2] = forward
        mat[0:3, 3] = cam_pos
        poses.append(mat.tolist())
    return poses

# Global singleton engine and execution lock
_ENGINE: Optional[OnlineReconstructionEngine] = None
_ENGINE_LOCK = threading.Lock()
_CURRENT_STREAM_CANCEL = threading.Event()

def get_engine() -> OnlineReconstructionEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            print("[Streaming Server] Initializing ABot-Recon Online Reconstruction Engine on GPU...", flush=True)
            _ENGINE = OnlineReconstructionEngine(
                checkpoint=ROOT_DIR / "checkpoints/abot_recon.safetensors",
                device="cuda",
                confidence_threshold=0.1,
                point_stride=4,
            )
            print("[Streaming Server] ABot-Recon Model Loaded Successfully.", flush=True)
        return _ENGINE


class StreamingRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()
    def do_HEAD(self) -> None:
        self.do_GET()


    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html", "/viewer"):
            self.path = "/viewer/index.html"
            return super().do_GET()

        if path == "/api/stream":
            return self.handle_sse_stream(parsed_url.query)
        if path == "/api/video_info":
            return self.handle_video_info(parsed_url.query)
        if path == "/api/frame":
            return self.handle_video_frame(parsed_url.query)
        if path == "/api/trajectory_info":
            return self.handle_trajectory_info(parsed_url.query)
        if path == "/api/sequences":
            return self.handle_sequences()
        if path in ("/api/offline_models", "/api/models"):
            return self.handle_offline_models()

        if path == "/api/status":
            return self.handle_status()
        if path == "/api/stop":
            return self.handle_stop_stream()
        return super().do_GET()

    def handle_offline_models(self) -> None:
        """List all reconstructed 3D point cloud models stored on disk in outputs/."""
        models = scan_all_ply_models()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(models, ensure_ascii=False).encode("utf-8"))
        return

    def handle_video_info(self, query_string: str) -> None:
        """Return video availability, frame count, and dimensions for a given model or directory."""
        params = urllib.parse.parse_qs(query_string)
        model_path_str = params.get("path", [""])[0].lstrip("/")
        if not model_path_str:
            self.send_error(400, "Missing path parameter")
            return

        target_path = ROOT_DIR / model_path_str
        colors = load_colors_tensor(target_path)
        if colors is not None:
            info = {
                "has_video": True,
                "total_frames": len(colors),
                "height": int(colors.shape[1]),
                "width": int(colors.shape[2]),
                "format": "colors.pt",
                "path": model_path_str,
            }
        else:
            img_dir = target_path if target_path.is_dir() else target_path.parent
            meta_file = img_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        raw_dir_name = meta.get("image_dir")
                        if raw_dir_name and (ROOT_DIR / raw_dir_name).is_dir():
                            img_dir = ROOT_DIR / raw_dir_name
                except Exception:
                    pass

            frames = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ) if img_dir.is_dir() else []

            if frames:
                info = {
                    "has_video": True,
                    "total_frames": len(frames),
                    "height": 0,
                    "width": 0,
                    "format": "images",
                    "path": str(img_dir.relative_to(ROOT_DIR)),
                }
            else:
                info = {
                    "has_video": False,
                    "total_frames": 0,
                    "format": "none",
                    "path": model_path_str,
                }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(info).encode("utf-8"))

    def handle_trajectory_info(self, query_string: str) -> None:
        """Return trajectory availability and URL for a given PLY model."""
        params = urllib.parse.parse_qs(query_string)
        model_path_str = params.get("path", [""])[0].lstrip("/")
        full_path = (ROOT_DIR / model_path_str).resolve()

        traj_url = None
        if full_path.is_file() and full_path.suffix.lower() == ".ply":
            traj_url = find_trajectory_for_ply(full_path)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "has_trajectory": traj_url is not None,
            "trajectory_url": traj_url,
        }).encode("utf-8"))

    def handle_video_frame(self, query_string: str) -> None:
        """Return a single JPEG image frame from colors.pt or raw image sequence."""
        params = urllib.parse.parse_qs(query_string)
        model_path_str = params.get("path", [""])[0].lstrip("/")
        frame_idx = int(params.get("frame", ["0"])[0])
        quality = int(params.get("quality", ["80"])[0])

        if not model_path_str:
            self.send_error(400, "Missing path parameter")
            return

        target_path = ROOT_DIR / model_path_str
        colors = load_colors_tensor(target_path)
        if colors is not None:
            if frame_idx < 0 or frame_idx >= len(colors):
                self.send_error(404, f"Frame {frame_idx} out of range (0-{len(colors)-1})")
                return
            img_rgb = colors[frame_idx].numpy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                self.send_error(500, "Failed to encode JPEG frame")
                return
            jpeg_bytes = buf.tobytes()
        else:
            img_dir = target_path if target_path.is_dir() else target_path.parent
            meta_file = img_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        raw_dir_name = meta.get("image_dir")
                        if raw_dir_name and (ROOT_DIR / raw_dir_name).is_dir():
                            img_dir = ROOT_DIR / raw_dir_name
                except Exception:
                    pass

            frames = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ) if img_dir.is_dir() else []

            if not frames or frame_idx < 0 or frame_idx >= len(frames):
                self.send_error(404, f"Frame {frame_idx} not found")
                return

            with open(frames[frame_idx], "rb") as f:
                jpeg_bytes = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg_bytes)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(jpeg_bytes)

    def handle_sequences(self) -> None:
        """Scan and list all available datasets: both outputs PLY models and raw video sequences."""
        sequences = []

        # 1. Output PLY point cloud data sources
        ply_models = scan_all_ply_models()
        for item in ply_models:
            sequences.append({
                "id": item["path"],
                "name": f"📁 [outputs 点云流] {item['name']}",
                "path": item["path"],
                "category": "📁 outputs 结果点云 (.ply 实时流式建图/回放)",
                "type": "ply",
                "frames": 60,
                "description": item["description"],
            })

        # 2. Dynamically scan outputs/**/frames for any extracted video frames
        outputs_dir = ROOT_DIR / "outputs"
        dynamic_frames = []
        if outputs_dir.is_dir():
            for frames_dir in sorted(outputs_dir.glob("**/frames"), key=lambda p: p.stat().st_mtime, reverse=True):
                if frames_dir.is_dir():
                    seq_name = frames_dir.parent.name
                    frames_list = sorted(frames_dir.glob("*.jpg"))
                    if frames_list:
                        rel_frames = str(frames_dir.relative_to(ROOT_DIR))
                        dynamic_frames.append({
                            "id": rel_frames,
                            "name": f"📹 视频抽帧序列 [{seq_name}] ({len(frames_list)} 帧)",
                            "path": rel_frames,
                            "category": "📹 视频抽帧数据源 (Extracted Video Frames)",
                            "type": "video",
                            "frames": len(frames_list),
                            "description": f"已抽帧图像目录: {rel_frames} ({len(frames_list)} 帧)",
                        })
        sequences = dynamic_frames + sequences

        # 3. Dynamically scan data/ and examples/ for any raw image sequence folders
        scan_dirs = [ROOT_DIR / "data", ROOT_DIR / "examples"]
        for base_dir in scan_dirs:
            if not base_dir.is_dir():
                continue
            for d in sorted(base_dir.glob("**"), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir() and "frames" not in d.parts:
                    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
                    if len(imgs) >= 10:
                        rel_path = str(d.relative_to(ROOT_DIR))
                        if not any(s.get("path") == rel_path for s in sequences):
                            sequences.append({
                                "id": rel_path,
                                "name": f"📹 原始图像序列 [{d.parent.name}/{d.name}] ({len(imgs)} 帧)",
                                "path": rel_path,
                                "category": "📹 原始图像序列 (GPU 在线推理建图)",
                                "type": "video",
                                "frames": len(imgs),
                                "description": f"图像序列: {rel_path} ({len(imgs)} 帧)",
                            })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(sequences, ensure_ascii=False).encode("utf-8"))
        return

    def handle_status(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        status = {
            "status": "ready",
            "model_loaded": _ENGINE is not None,
            "device": "cuda",
        }
        self.wfile.write(json.dumps(status).encode("utf-8"))
    def handle_stop_stream(self) -> None:
        """Signal any running streaming worker to abort immediately."""
        global _CURRENT_STREAM_CANCEL
        _CURRENT_STREAM_CANCEL.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))


    def handle_sse_stream(self, query_string: str) -> None:
        """Stream frame-by-frame 3D reconstruction using Server-Sent Events (SSE)."""
        global _CURRENT_STREAM_CANCEL

        # Preempt: Signal any currently running stream to abort immediately
        _CURRENT_STREAM_CANCEL.set()

        # Create a fresh cancellation event for this new stream session
        cancel_event = threading.Event()
        _CURRENT_STREAM_CANCEL = cancel_event

        params = urllib.parse.parse_qs(query_string)
        point_stride = int(params.get("point_stride", ["4"])[0])
        conf_thresh = float(params.get("confidence_threshold", ["0.1"])[0])
        frame_stride = int(params.get("stride", ["1"])[0])
        image_dir_name = params.get("sequence", [params.get("image_dir", ["examples/images"])[0]])[0]
        max_frames = int(params.get("max_frames", ["0"])[0])

        target_path = ROOT_DIR / image_dir_name
        if target_path.is_file() and target_path.suffix.lower() == ".ply":
            return self.handle_ply_sse_stream(target_path, params, cancel_event)

        if not target_path.is_dir():
            self.send_error(404, f"Image directory or PLY file {image_dir_name} not found")
            return
        image_dir = target_path

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )[::frame_stride]

        if max_frames > 0:
            image_paths = image_paths[:max_frames]

        if not image_paths:
            self.send_error(404, "No image frames found in directory")
            return

        print(f"[Streaming Server] Waiting to acquire engine lock for: {image_dir_name} ({len(image_paths)} frames)...", flush=True)

        engine = get_engine()
        with _ENGINE_LOCK:
            if cancel_event.is_set():
                print(f"[Streaming Server] Stream aborted before acquiring lock.", flush=True)
                return

            print(f"[Streaming Server] Lock acquired. Starting SSE stream: {image_dir_name} ({len(image_paths)} frames, stride={frame_stride})...", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def send_sse(event_name: str, payload: dict) -> bool:
                try:
                    data = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
                    self.wfile.write(data.encode("utf-8"))
                    self.wfile.flush()
                    return True
                except Exception:
                    return False

            # Send start event AFTER acquiring the engine lock so the client is in sync!
            if not send_sse("start", {
                "total_frames": len(image_paths),
                "point_stride": point_stride,
                "confidence_threshold": conf_thresh,
            }):
                return

            engine.reset()
            total_points = 0
            for i, img_path in enumerate(image_paths):
                if cancel_event.is_set():
                    print(f"[Streaming Server] Stream cancelled by user at frame {i+1}.", flush=True)
                    break

                res = engine.process_frame(
                    img_path,
                    confidence_threshold=conf_thresh,
                    point_stride=point_stride,
                    include_thumbnail=True,
                )
                total_points += res["point_count"]

                frame_data = {
                    "frame_index": res["frame_index"],
                    "total_frames": len(image_paths),
                    "image_url": f"/{img_path.relative_to(ROOT_DIR)}",
                    "thumbnail": res.get("thumbnail"),
                    "camera_pose": res["camera_pose"],
                    "point_count": res["point_count"],
                    "total_accumulated_points": total_points,
                    "points_xyz": res["points_xyz"].tolist(),
                    "points_rgb": res["points_rgb"].tolist(),
                    "latency_ms": round(res["inference_time_ms"], 1),
                    "fps": round(res["fps"], 1),
                }

                print(
                    f"[Stream] Frame {res['frame_index']+1:02d}/{len(image_paths)} | "
                    f"Points: +{res['point_count']:,} (Total: {total_points:,}) | "
                    f"Latency: {res['inference_time_ms']:.1f}ms ({res['fps']:.1f} FPS)",
                    flush=True,
                )

                if not send_sse("frame", frame_data):
                    print(f"[Streaming Server] Client disconnected at frame {i+1}.", flush=True)
                    break

                time.sleep(0.01)

            if not cancel_event.is_set():
                send_sse("complete", {
                    "total_frames": len(image_paths),
                    "total_points": total_points,
                })
                print(f"[Streaming Server] Completed stream of {len(image_paths)} frames.", flush=True)

    def handle_ply_sse_stream(self, ply_path: Path, params: dict, cancel_event: threading.Event) -> None:
        """Stream progressive 3D point cloud chunks frame-by-frame via SSE with real video stream."""
        point_stride = int(params.get("point_stride", ["4"])[0])
        max_frames_param = int(params.get("max_frames", ["0"])[0])
        conf_thresh = float(params.get("confidence_threshold", ["0.1"])[0])

        colors_tensor = load_colors_tensor(ply_path)
        wp_path = ply_path.parent / "world_points.pt"
        conf_path = ply_path.parent / "confidence.pt"
        has_video = colors_tensor is not None

        poses = None
        pose_candidates = [
            ply_path.with_name(f"{ply_path.stem}_poses.npy"),
            ply_path.with_name(f"{ply_path.stem}_camera_poses.npy"),
        ]
        if "reconstruction" in ply_path.name.lower():
            pose_candidates.extend([
                ply_path.parent / "camera_poses.npy",
                ply_path.parent / "camera_poses_loop.npy",
                ply_path.parent / "camera_poses_noloop.npy",
            ])
        for p_cand in pose_candidates:
            if p_cand.is_file():
                try:
                    loaded_poses = np.load(p_cand).astype(np.float32)
                    if loaded_poses.ndim == 3 and loaded_poses.shape[1:] == (4, 4):
                        poses = loaded_poses
                        break
                    elif loaded_poses.ndim == 2 and loaded_poses.shape[1] == 16:
                        poses = loaded_poses.reshape(-1, 4, 4)
                        break
                except Exception:
                    pass

        has_world_points = wp_path.is_file() and has_video
        world_points_tensor = None
        conf_tensor = None
        if has_world_points:
            try:
                world_points_tensor = torch.load(wp_path, map_location="cpu", weights_only=False)
                if conf_path.is_file():
                    conf_tensor = torch.load(conf_path, map_location="cpu", weights_only=False)
                print(f"[Streaming Server] Loaded full per-frame tensors for {ply_path.name}: {world_points_tensor.shape[0]} frames.", flush=True)
            except Exception as e:
                print(f"[Streaming Server] Failed to load world_points.pt: {e}, falling back to PLY file.", flush=True)
                has_world_points = False

        if not has_world_points:
            print(f"[Streaming Server] Loading PLY file for SSE streaming: {ply_path}...", flush=True)
            try:
                pcd = o3d.io.read_point_cloud(str(ply_path))
                points_xyz = np.asarray(pcd.points, dtype=np.float32)
                points_rgb = np.asarray(pcd.colors, dtype=np.float32)
            except Exception as e:
                self.send_error(500, f"Failed to parse PLY file {ply_path}: {e}")
                return

            if len(points_xyz) == 0:
                self.send_error(400, "Point cloud is empty")
                return

            if len(points_rgb) == len(points_xyz):
                if points_rgb.max() <= 1.01:
                    points_rgb = (points_rgb * 255.0).clip(0, 255).astype(np.int32)
                else:
                    points_rgb = points_rgb.clip(0, 255).astype(np.int32)
            else:
                points_rgb = np.full_like(points_xyz, 204, dtype=np.int32)

            if point_stride > 1:
                points_xyz = points_xyz[::point_stride]
                points_rgb = points_rgb[::point_stride]

        if has_world_points:
            num_available_frames = min(len(world_points_tensor), len(colors_tensor))
            if max_frames_param > 0:
                total_frames = min(max_frames_param, num_available_frames)
            else:
                total_frames = num_available_frames
        else:
            if max_frames_param > 0:
                total_frames = max_frames_param
            elif has_video:
                total_frames = min(len(colors_tensor), 120)
            elif poses is not None and len(poses) > 0:
                total_frames = min(len(poses), 120)
            else:
                total_frames = 60

        total_frames = max(1, total_frames)

        if poses is None or len(poses) == 0:
            if not has_world_points:
                poses_list = generate_orbital_poses(points_xyz, total_frames)
            else:
                poses_list = [np.eye(4, dtype=np.float32).tolist() for _ in range(total_frames)]
        else:
            poses_list = []
            for i in range(total_frames):
                p_idx = int(i * len(poses) / total_frames)
                poses_list.append(poses[p_idx].tolist())

        video_desc = f"with colors.pt video stream ({len(colors_tensor)} frames)" if has_video else "without colors.pt"
        print(f"[Streaming Server] Starting PLY SSE stream: {ply_path.name} ({total_frames} frames, {video_desc})...", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send_sse(event_name: str, payload: dict) -> bool:
            try:
                data = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()
                return True
            except Exception:
                return False

        if not send_sse("start", {
            "total_frames": total_frames,
            "point_stride": point_stride,
            "confidence_threshold": conf_thresh,
        }):
            return

        if not has_world_points:
            chunk_size = int(np.ceil(len(points_xyz) / total_frames))

        rel_path = str(ply_path.relative_to(ROOT_DIR))
        total_accumulated = 0

        for i in range(total_frames):
            if cancel_event.is_set():
                print(f"[Streaming Server] PLY stream cancelled by user at frame {i+1}.", flush=True)
                break

            c_idx = i if has_world_points else int(i * len(colors_tensor) / total_frames) if has_video else 0
            if has_video:
                hud_thumb = encode_frame_jpeg_base64(colors_tensor, c_idx)
                hud_url = f"/api/frame?path={urllib.parse.quote(rel_path)}&frame={c_idx}"
            else:
                hud_thumb = None
                hud_url = "/examples/images/000000.png"

            if has_world_points:
                step = max(1, point_stride)
                wp_f = world_points_tensor[c_idx, ::step, ::step].reshape(-1, 3).numpy()
                col_f = colors_tensor[c_idx, ::step, ::step].reshape(-1, 3).numpy()
                valid = ~np.isnan(wp_f).any(axis=-1) & ~np.isinf(wp_f).any(axis=-1)
                if conf_tensor is not None:
                    c_mask = conf_tensor[c_idx, ::step, ::step].reshape(-1).numpy() >= conf_thresh
                    valid &= c_mask
                chunk_xyz = wp_f[valid]
                chunk_rgb = col_f[valid].astype(np.int32)
            else:
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(points_xyz))
                chunk_xyz = points_xyz[start_idx:end_idx]
                chunk_rgb = points_rgb[start_idx:end_idx]

            total_accumulated += len(chunk_xyz)

            frame_data = {
                "frame_index": i,
                "total_frames": total_frames,
                "image_url": hud_url,
                "thumbnail": hud_thumb,
                "camera_pose": poses_list[i],
                "point_count": len(chunk_xyz),
                "total_accumulated_points": total_accumulated,
                "points_xyz": chunk_xyz.round(4).tolist(),
                "points_rgb": chunk_rgb.tolist(),
                "latency_ms": 10.0,
                "fps": 30.0,
            }

            if not send_sse("frame", frame_data):
                print(f"[Streaming Server] Client disconnected at frame {i+1}.", flush=True)
                break

            time.sleep(0.02)

        if not cancel_event.is_set():
            send_sse("complete", {
                "total_frames": total_frames,
                "total_points": total_accumulated,
            })
            print(f"[Streaming Server] Completed PLY stream of {total_frames} frames ({total_accumulated:,} points).", flush=True)
        pass


def run_server(port: int = 8088, host: str = "0.0.0.0") -> None:
    server_address = (host, port)
    ThreadingHTTPServer.allow_reuse_address = True

    # Pre-warm model and compile JIT kernels on GPU
    engine = get_engine()
    sample_imgs = sorted(Path("examples/images").glob("*.png"))
    if sample_imgs:
        print("[Streaming Server] Pre-warming GPU kernels...", flush=True)
        engine.process_frame(sample_imgs[0])
        engine.reset()
        print("[Streaming Server] GPU kernels warmed up and ready.", flush=True)

    httpd = ThreadingHTTPServer(server_address, StreamingRequestHandler)
    print(f"[ABot-Recon 3D Streaming Server] Running at http://127.0.0.1:{port}/", flush=True)
    print(f"[ABot-Recon 3D Streaming Server] Access from browser: http://127.0.0.1:{port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ABot-Recon 3D streaming visualizer")
    parser.add_argument("--port", type=int, default=8088, help="Port to serve on (default: 8088)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
