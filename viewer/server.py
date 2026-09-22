#!/usr/bin/env python3
"""Multi-threaded HTTP & Server-Sent Events (SSE) Streaming Server for ABot-Recon 3D Visualizer."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
import email.parser
import asyncio
import subprocess
import shutil
import websockets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import base64
import cv2
import numpy as np
import open3d as o3d
import torch
import struct
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from viewer.stream_backend import OnlineReconstructionEngine

# In-memory cache for colors.pt video stream tensors (max 8 models)
_COLORS_CACHE: Dict[str, torch.Tensor] = {}
# In-memory cache for dynamic replay buffers (max 6 models)
_REPLAY_CACHE: Dict[str, bytes] = {}
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

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
        parent / "camera_poses.npy",
        parent / "camera_poses_loop.npy",
        parent / "camera_poses_noloop.npy",
        parent / "robot_poses.npy",
    ]
    if parent.is_dir():
        for p in parent.glob("*-robot_poses.npy"):
            candidates.append(p)
        for p in parent.glob("*camera_poses*.npy"):
            candidates.append(p)
        for p in parent.glob("*poses*.npy"):
            candidates.append(p)

    for c in candidates:
        if c.is_file():
            try:
                rel = c.relative_to(ROOT_DIR)
                return f"/{rel}"
            except ValueError:
                pass
    return None

def generate_replay_buffer(target_path: Path, stride: int = 4, conf_thresh: float = 0.1) -> bytes:
    """Generate binary buffer containing per-frame points and colors for dynamic mapping playback."""
    target_path = target_path.resolve()
    model_dir = target_path if target_path.is_dir() else target_path.parent

    wp_path = model_dir / "world_points.pt"
    cp_path = model_dir / "colors.pt"
    conf_path = model_dir / "confidence.pt"

    frame_counts: List[int] = []
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []

    if wp_path.is_file() and cp_path.is_file():
        wp = torch.load(wp_path, map_location="cpu", weights_only=False)
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        conf = torch.load(conf_path, map_location="cpu", weights_only=False) if conf_path.is_file() else None

        total_frames = min(len(wp), len(cp))
        step = max(1, stride)
        for i in range(total_frames):
            w_f = wp[i, ::step, ::step].reshape(-1, 3).numpy()
            c_f = cp[i, ::step, ::step].reshape(-1, 3).numpy()
            valid = np.isfinite(w_f).all(axis=-1)
            if conf is not None:
                cf = conf[i, ::step, ::step].reshape(-1).numpy()
                valid &= (cf >= conf_thresh)
            xyz_v = w_f[valid].astype(np.float32)
            rgb_v = c_f[valid].astype(np.uint8)
            frame_counts.append(len(xyz_v))
            all_xyz.append(xyz_v)
            all_rgb.append(rgb_v)
    else:
        # Fallback to PLY file + poses
        ply_path = target_path if target_path.is_file() and target_path.suffix.lower() == ".ply" else model_dir / "reconstruction.ply"
        if not ply_path.is_file():
            ply_files = list(model_dir.glob("*.ply"))
            if ply_files:
                ply_path = ply_files[0]
            else:
                raise FileNotFoundError(f"No PLY file found in {model_dir}")

        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pcd.colors) == len(pts):
            c_arr = np.asarray(pcd.colors, dtype=np.float32)
            if c_arr.max() <= 1.01:
                colors = (c_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                colors = c_arr.clip(0, 255).astype(np.uint8)
        else:
            colors = np.full((len(pts), 3), 200, dtype=np.uint8)

        # Find poses
        poses = None
        traj_url = find_trajectory_for_ply(ply_path)
        if traj_url:
            traj_path = ROOT_DIR / traj_url.lstrip("/")
            if traj_path.is_file():
                try:
                    loaded = np.load(traj_path).astype(np.float32)
                    if loaded.ndim == 3 and loaded.shape[1:] == (4, 4):
                        poses = loaded
                    elif loaded.ndim == 2 and loaded.shape[1] == 16:
                        poses = loaded.reshape(-1, 4, 4)
                except Exception:
                    pass

        # Downsample if too many points (> 1.2M points)
        if len(pts) > 1200000:
            sub_step = int(np.ceil(len(pts) / 1200000))
            pts = pts[::sub_step]
            colors = colors[::sub_step]

        if poses is not None and len(poses) > 0 and cKDTree is not None:
            total_frames = len(poses)
            cam_centers = poses[:, :3, 3]
            tree = cKDTree(cam_centers)
            _, assignments = tree.query(pts, workers=-1)
            sort_idx = np.argsort(assignments)
            pts = pts[sort_idx]
            colors = colors[sort_idx]
            counts = np.bincount(assignments, minlength=total_frames)
            frame_counts = counts.tolist()
            all_xyz = [pts]
            all_rgb = [colors]
        else:
            total_frames = len(poses) if poses is not None else 60
            chunk_size = int(np.ceil(len(pts) / total_frames))
            frame_counts = []
            for i in range(total_frames):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, len(pts))
                frame_counts.append(max(0, e - s))
            all_xyz = [pts]
            all_rgb = [colors]

    if all_xyz:
        xyz_arr = np.concatenate(all_xyz, axis=0)
        rgb_arr = np.concatenate(all_rgb, axis=0)
    else:
        xyz_arr = np.zeros((0, 3), dtype=np.float32)
        rgb_arr = np.zeros((0, 3), dtype=np.uint8)

    total_points = len(xyz_arr)
    header = struct.pack("<4sIII", b"ABTR", 1, total_frames, total_points)
    fc_bytes = np.array(frame_counts, dtype=np.uint32).tobytes()
    xyz_bytes = xyz_arr.astype(np.float32).tobytes()
    rgb_bytes = rgb_arr.astype(np.uint8).tobytes()

    return header + fc_bytes + xyz_bytes + rgb_bytes

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
        "✨ 3D 高斯泼溅 (3D Gaussian Splatting)": -1,
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
        try:
            rel_dir = str(ply_path.parent.relative_to(ROOT_DIR))
        except ValueError:
            rel_dir = str(ply_path.parent.name)
        stem = ply_path.stem
        fname = ply_path.name
        # 0. 3D Gaussian Splatting (3DGS)
        if "3dgs" in stem or "splat" in stem:
            category = "✨ 3D 高斯泼溅 (3D Gaussian Splatting)"
            tag = "✨ [3DGS 高斯]"
            name = f"{tag} {stem} ({format_point_count(vertex_count)} 高斯, {size_mb:.1f}MB, {mtime_str})"
        # 1. Multi-Stream / General Fusion
        elif "fusion" in rel_str or "merged" in stem or "fused_" in stem:
            category = "🔥 多视角融合点云 (Multi-Stream Fusion)"
            is_colored = "colored" in stem
            is_full = "full" in stem
            tag = "🎨 [区分色彩]" if is_colored else "🌟 [正常真彩]"
            mode_str = "全量无损" if is_full else "体素去重"
            if "fused_" in stem:
                parts = stem.split("_")
                num_streams = f"{parts[1]}路" if len(parts) > 1 and parts[1].isdigit() else ""
                name = f"{tag} {num_streams}融合-{mode_str} ({fname}, {format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
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
            name = f"{tag} {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 3. Single Video Reconstructions
        elif fname == "reconstruction.ply" or (ply_path.parent / "metadata.json").is_file():
            category = "📹 单视频重建点云 (Single Video)"
            name = f"📹 [单视频重建] {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 4. Multi-Stream Scenes
        elif "scenes" in rel_str:
            category = "🏛️ 场景全景融合 (Scene Fusion)"
            is_colored = "colored" in stem
            tag = "🎨 [场景多色]" if is_colored else "🏛️ [场景全景]"
            name = f"{tag} {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 5. Streaming Sessions
        elif "streams" in rel_str:
            category = "📡 实时流式会话 (Streaming Sessions)"
            name = f"📡 [流式会话] {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 6. Loop Comparison
        elif "loop" in rel_str:
            category = "🔄 回环与轨迹对比 (Loop Comparison)"
            tag = "🔄 [回环优化]" if "loop" in stem else "🚀 [原始流式]"
            name = f"{tag} {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        # 7. Other Arbitrary Outputs
        else:
            category = "📁 输出点云 (Outputs Point Clouds)"
            name = f"📁 {fname} ({format_point_count(vertex_count)}, {size_mb:.1f}MB, {mtime_str})"
        description = f"路径: {rel_str} | 大小: {size_mb:.2f}MB | 点数: {vertex_count:,} | 时间: {mtime_str}" if vertex_count else f"路径: {rel_str} | 大小: {size_mb:.2f}MB | 时间: {mtime_str}"

        models.append({
            "url": url,
            "path": rel_str,
            "rel_dir": rel_dir,
            "filename": fname,
            "name": name,
            "category": category,
            "description": description,
            "size_bytes": size_bytes,
            "vertex_count": vertex_count,
            "mtime": mtime,
            "trajectory_url": find_trajectory_for_ply(ply_path),
            "_prio": category_priority.get(category, 50),
        })

    # Sort models: Primary key is category priority, Secondary is rel_dir (folder group), Tertiary is mtime descending
    models.sort(key=lambda m: (m["_prio"], m["rel_dir"], -m["mtime"], m["name"]))
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
_DEVICE: str = "cuda"

def get_engine(device: Optional[str] = None) -> OnlineReconstructionEngine:
    global _ENGINE, _DEVICE
    if device is not None:
        _DEVICE = device
    with _ENGINE_LOCK:
        if _ENGINE is None:
            print(f"[Streaming Server] Initializing ABot-Recon Online Reconstruction Engine on {_DEVICE}...", flush=True)
            _ENGINE = OnlineReconstructionEngine(
                checkpoint=ROOT_DIR / "checkpoints/abot_recon.safetensors",
                device=_DEVICE,
                confidence_threshold=0.1,
                point_stride=4,
            )
            print("[Streaming Server] ABot-Recon Model Loaded Successfully.", flush=True)
        return _ENGINE

# ---------------------------------------------------------------------------
# Video Reconstruction Task Management & 8090 Streaming Bridge
# ---------------------------------------------------------------------------
_VIDEO_TASKS: Dict[str, Dict[str, Any]] = {}
_VIDEO_TASKS_LOCK = threading.Lock()
_LAST_8090_SESSIONS: Dict[str, Any] = {"status": "stopped", "active_sessions": [], "completed_sessions": []}

def is_port_listening(port: int = 8090, host: str = "127.0.0.1") -> bool:
    """Quickly check if TCP port is listening with robust 2.5s timeout."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2.5):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def ensure_streaming_service() -> bool:
    """Ensure that the 8090 streaming reconstruction engine is active and responding."""
    # 1. Quick check: if port is already listening, verify readiness
    if is_port_listening(8090):
        for _ in range(6):
            try:
                req = urllib.request.Request("http://127.0.0.1:8090/")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(0.5)
        # If TCP connects, 8090 daemon is running and alive
        return True

    # 2. Port not listening, attempt to start 8090
    print("[Video Service] 8090 Streaming API is not listening. Spawning background service...", flush=True)
    cmd = [sys.executable, str(ROOT_DIR / "viewer/streaming_api_server.py"), "--host", "0.0.0.0", "--port", "8090"]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[Video Service] Failed to spawn 8090 service: {e}", flush=True)
        return False

    for _ in range(40):
        time.sleep(0.5)
        if is_port_listening(8090):
            try:
                req = urllib.request.Request("http://127.0.0.1:8090/")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        print("[Video Service] 8090 Streaming API is now ready!", flush=True)
                        return True
            except Exception:
                pass
    return is_port_listening(8090)

def parse_multipart_request(headers: Any, rfile: Any, content_length: int) -> Tuple[Dict[str, str], Optional[Path], str]:
    """Parse multipart/form-data directly, streaming any file part to outputs/uploads/."""
    upload_dir = ROOT_DIR / "outputs" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    content_type = headers.get("Content-Type", "")
    fields: Dict[str, str] = {}
    saved_file_path: Optional[Path] = None
    saved_filename = ""

    if content_type.startswith("multipart/form-data"):
        raw_body = rfile.read(content_length)
        header_bytes = b"Content-Type: " + content_type.encode("latin1") + b"\r\n\r\n"
        msg = email.parser.BytesParser().parsebytes(header_bytes + raw_body)

        if msg.is_multipart():
            for part in msg.get_payload():
                cd = part.get("Content-Disposition")
                if not cd:
                    continue
                name = part.get_param("name", header="Content-Disposition")
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                if filename and payload:
                    clean_name = Path(filename).name.replace(" ", "_")
                    t_prefix = time.strftime("%Y%m%d_%H%M%S")
                    target_path = upload_dir / f"{t_prefix}_{clean_name}"
                    with open(target_path, "wb") as f:
                        f.write(payload)
                    saved_file_path = target_path
                    saved_filename = filename
                elif name and payload:
                    fields[name] = payload.decode("utf-8", errors="ignore")
    elif content_length > 0:
        # Fallback: treat entire body as raw video data
        t_prefix = time.strftime("%Y%m%d_%H%M%S")
        target_path = upload_dir / f"{t_prefix}_upload.mp4"
        with open(target_path, "wb") as f:
            remaining = content_length
            while remaining > 0:
                chunk_size = min(remaining, 1024 * 1024)
                chunk = rfile.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        saved_file_path = target_path
        saved_filename = "upload.mp4"

    return fields, saved_file_path, saved_filename

async def _stream_video_frames_to_8090(
    task_id: str,
    video_path: Path,
    scene_id: str,
    session_id: str,
    point_stride: int,
    frame_stride: int,
    voxel_size: float,
    confidence_threshold: float,
    auto_fuse: bool,
    dynamic_filter: bool,
) -> Dict[str, Any]:
    ws_url = (
        f"ws://127.0.0.1:8090/ws/stream?"
        f"session_id={urllib.parse.quote(session_id)}&"
        f"scene_id={urllib.parse.quote(scene_id)}&"
        f"point_stride={point_stride}&"
        f"frame_stride={frame_stride}&"
        f"voxel_size={voxel_size}&"
        f"confidence_threshold={confidence_threshold}&"
        f"auto_fuse={'true' if auto_fuse else 'false'}&"
        f"dynamic_filter={'true' if dynamic_filter else 'false'}&"
        f"include_points=false"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法读取视频文件: {video_path.name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_total = max(1, total_frames // frame_stride) if total_frames > 0 else 0

    with _VIDEO_TASKS_LOCK:
        if task_id in _VIDEO_TASKS:
            _VIDEO_TASKS[task_id].update({
                "status": "reconstructing",
                "total_frames": total_frames,
                "effective_frames": effective_total,
                "video_fps": round(video_fps, 1),
                "resolution": f"{width}x{height}",
                "message": f"连接 8090 神经建图引擎 (视频共 {total_frames} 帧, 抽帧步长 {frame_stride})...",
                "progress": 5.0,
            })

    async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
        handshake_raw = await ws.recv()
        handshake = json.loads(handshake_raw)

        with _VIDEO_TASKS_LOCK:
            if task_id in _VIDEO_TASKS:
                _VIDEO_TASKS[task_id]["message"] = "8090 引擎就绪，正在逐帧流式推理与自回归位姿跟踪..."
                _VIDEO_TASKS[task_id]["progress"] = 10.0

        frame_idx = 0
        sent_count = 0
        t0_stream = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success:
                    await ws.send(buf.tobytes())
                    sent_count += 1

                    ack_raw = await ws.recv()
                    ack = json.loads(ack_raw)

                    if effective_total > 0:
                        infer_pct = min(70.0, (sent_count / effective_total) * 70.0)
                    else:
                        infer_pct = min(70.0, sent_count * 0.5)

                    current_progress = round(10.0 + infer_pct, 1)
                    now = time.time()
                    elapsed = now - t0_stream
                    current_fps = round(sent_count / elapsed, 1) if elapsed > 0 else 0.0

                    with _VIDEO_TASKS_LOCK:
                        if task_id in _VIDEO_TASKS:
                            _VIDEO_TASKS[task_id].update({
                                "current_frame": sent_count,
                                "progress": current_progress,
                                "fps": current_fps,
                                "message": f"3D 神经流式建图中: 帧 {sent_count}/{effective_total} ({current_progress}%, {current_fps} FPS)",
                                "updated_at": now,
                            })

            frame_idx += 1

        cap.release()

        with _VIDEO_TASKS_LOCK:
            if task_id in _VIDEO_TASKS:
                _VIDEO_TASKS[task_id].update({
                    "status": "finalizing",
                    "progress": 82.0,
                    "message": "视频所有帧已推送完毕，正在进行空间体素滤波去重与三维点云固化...",
                    "updated_at": time.time(),
                })

        await ws.send(json.dumps({"type": "EOS"}))

        if auto_fuse:
            with _VIDEO_TASKS_LOCK:
                if task_id in _VIDEO_TASKS:
                    _VIDEO_TASKS[task_id].update({
                        "progress": 88.0,
                        "message": "正在执行跨视角多模态配准与 optv2 全局几何与点云优化 (PGO + MLS + 去噪)...",
                        "updated_at": time.time(),
                    })

        summary_raw = await ws.recv()
        summary = json.loads(summary_raw)
        return summary

def run_video_pipeline_thread(
    task_id: str,
    video_path: Path,
    scene_id: str,
    session_id: str,
    point_stride: int,
    frame_stride: int,
    voxel_size: float,
    confidence_threshold: float,
    auto_fuse: bool,
    dynamic_filter: bool,
) -> None:
    try:
        with _VIDEO_TASKS_LOCK:
            if task_id in _VIDEO_TASKS:
                _VIDEO_TASKS[task_id]["message"] = "检查 8090 神经建图推理服务状态..."
                _VIDEO_TASKS[task_id]["progress"] = 3.0

        if not ensure_streaming_service():
            raise RuntimeError("8090 实时流式建图服务未启动且自动拉起超时，请先检查服务状态")

        summary = asyncio.run(
            _stream_video_frames_to_8090(
                task_id=task_id,
                video_path=video_path,
                scene_id=scene_id,
                session_id=session_id,
                point_stride=point_stride,
                frame_stride=frame_stride,
                voxel_size=voxel_size,
                confidence_threshold=confidence_threshold,
                auto_fuse=auto_fuse,
                dynamic_filter=dynamic_filter,
            )
        )

        scan_all_ply_models(force=True)

        clean_scene_id = summary.get("scene_id", scene_id)
        folder_name = summary.get("folder_name", session_id)

        scene_disk_ply = ROOT_DIR / "outputs" / "scenes" / clean_scene_id / "reconstruction.ply"
        stream_disk_ply = ROOT_DIR / "outputs" / "streams" / folder_name / "reconstruction.ply"

        if scene_disk_ply.is_file():
            viewer_url = f"/outputs/scenes/{clean_scene_id}/reconstruction.ply"
        elif stream_disk_ply.is_file():
            viewer_url = f"/outputs/streams/{folder_name}/reconstruction.ply"
        else:
            viewer_url = summary.get("scene_download_url") or summary.get("deliverables", {}).get("ply_url")

        with _VIDEO_TASKS_LOCK:
            if task_id in _VIDEO_TASKS:
                _VIDEO_TASKS[task_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "message": "🎉 3D 建图与优化已全部完成！",
                    "result": summary,
                    "viewer_url": viewer_url,
                    "scene_id": clean_scene_id,
                    "folder_name": folder_name,
                    "dedup_point_count": summary.get("dedup_point_count", 0),
                    "updated_at": time.time(),
                })
        print(f"[Video Pipeline] Task [{task_id}] finished successfully! Viewer URL: {viewer_url}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _VIDEO_TASKS_LOCK:
            if task_id in _VIDEO_TASKS:
                _VIDEO_TASKS[task_id].update({
                    "status": "failed",
                    "error": str(e),
                    "message": f"建图失败: {e}",
                    "updated_at": time.time(),
                })

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

        if path in ("/", "/index.html", "/viewer", "/viewer/index.html"):
            index_path = ROOT_DIR / "viewer" / "index.html"
            if index_path.is_file():
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(content)
                return
        if path in ("/gaussian.html", "/splat", "/gaussian", "/viewer/gaussian.html"):
            gauss_path = ROOT_DIR / "viewer" / "gaussian.html"
            if gauss_path.is_file():
                with open(gauss_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(content)
                return
        if path == "/api/stream":
            return self.handle_sse_stream(parsed_url.query)
        if path == "/api/video_info":
            return self.handle_video_info(parsed_url.query)
        if path == "/api/replay_pointcloud":
            return self.handle_replay_pointcloud(parsed_url.query)
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

        # Video Upload & 3D Reconstruction Endpoints (8090 Pipeline Bridge)
        if path in ("/api/video/status", "/api/upload/status"):
            return self.handle_video_status(parsed_url.query)
        if path in ("/api/video/tasks", "/api/upload/tasks"):
            return self.handle_video_tasks()
        if path in ("/api/video/progress", "/api/upload/progress"):
            return self.handle_video_progress_sse(parsed_url.query)
        if path == "/api/scenes":
            return self.handle_scenes_list()

        # 8090 Real-time Stream Relay Endpoints
        if path in ("/api/8090/sessions", "/api/live_streams"):
            return self.handle_8090_sessions()
        if path in ("/api/8090/live", "/api/stream/8090"):
            return self.handle_8090_live_sse(parsed_url.query)
        if path == "/api/8090/status":
            return self.handle_8090_status()
        if path == "/api/8090/snapshot":
            return self.handle_8090_snapshot(parsed_url.query)

        return super().do_GET()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/api/video/upload", "/api/upload_video", "/api/video/reconstruct"):
            return self.handle_video_upload(parsed_url.query)
        if path == "/api/stop":
            return self.handle_stop_stream()
        if path == "/api/convert_to_3dgs":
            return self.handle_convert_to_3dgs(parsed_url.query)
        self.send_error(404, f"POST endpoint not found: {path}")
    def handle_convert_to_3dgs(self, query_string: str) -> None:
        """Convert a given PLY point cloud into standard 3DGS Gaussian PLY & .splat format."""
        params = urllib.parse.parse_qs(query_string)
        model_path_str = params.get("path", [""])[0].lstrip("/")
        if not model_path_str:
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len > 0:
                    body = json.loads(self.rfile.read(content_len).decode("utf-8"))
                    model_path_str = body.get("path", "").lstrip("/")
            except Exception:
                pass

        if not model_path_str:
            self.send_error(400, "Missing path parameter")
            return

        target_ply = ROOT_DIR / model_path_str
        if not target_ply.is_file():
            self.send_error(404, f"Target point cloud file not found: {model_path_str}")
            return

        from scripts.pcd_to_3dgs import convert_point_cloud_to_3dgs

        try:
            out_ply = target_ply.parent / f"{target_ply.stem}_3dgs.ply"
            res = convert_point_cloud_to_3dgs(
                input_path=target_ply,
                output_path=out_ply,
                scale_mode="adaptive",
                base_scale=0.006,
                thin_factor=0.15,
                opacity=0.95,
                sh_degree=0,
                export_splat=True,
            )
            scan_all_ply_models(force=True)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            rel_3dgs_ply = str(out_ply.relative_to(ROOT_DIR))
            res["viewer_url"] = f"/{rel_3dgs_ply}"
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"3DGS conversion failed: {e}")


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

        model_dir = target_path if target_path.is_dir() else target_path.parent
        has_wp = (model_dir / "world_points.pt").is_file() and (model_dir / "colors.pt").is_file()
        traj_url = find_trajectory_for_ply(target_path) if target_path.is_file() and target_path.suffix.lower() == ".ply" else None
        has_replay = has_wp or (traj_url is not None)
        info["has_replay"] = has_replay
        info["replay_url"] = f"/api/replay_pointcloud?path={urllib.parse.quote(model_path_str)}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(info).encode("utf-8"))

    def handle_replay_pointcloud(self, query_string: str) -> None:
        """Return binary stream containing per-frame point cloud for dynamic mapping replay."""
        global _REPLAY_CACHE
        params = urllib.parse.parse_qs(query_string)
        model_path_str = params.get("path", [""])[0].lstrip("/")
        stride = int(params.get("stride", ["4"])[0])
        conf_thresh = float(params.get("confidence_threshold", ["0.1"])[0])

        if not model_path_str:
            self.send_error(400, "Missing path parameter")
            return

        target_path = ROOT_DIR / model_path_str
        if not target_path.exists():
            self.send_error(404, f"File or directory not found: {model_path_str}")
            return

        cache_key = f"{model_path_str}_{stride}_{conf_thresh}"
        if cache_key in _REPLAY_CACHE:
            payload = _REPLAY_CACHE[cache_key]
        else:
            try:
                payload = generate_replay_buffer(target_path, stride=stride, conf_thresh=conf_thresh)
                if len(_REPLAY_CACHE) > 6:
                    _REPLAY_CACHE.pop(next(iter(_REPLAY_CACHE)))
                _REPLAY_CACHE[cache_key] = payload
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Failed to generate replay pointcloud: {e}")
                return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

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
                "rel_dir": item.get("rel_dir", ""),
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
                            "rel_dir": str(frames_dir.parent.relative_to(ROOT_DIR)),
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
                                "rel_dir": str(d.parent.relative_to(ROOT_DIR)),
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
            "device": _DEVICE,
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

    def handle_video_upload(self, query_string: str) -> None:
        """Receive uploaded video file and trigger 8090 causal 3D reconstruction and optimization."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing video payload (Content-Length is 0)"}).encode("utf-8"))
            return

        fields, saved_file_path, original_filename = parse_multipart_request(self.headers, self.rfile, content_length)

        # Merge URL query parameters into fields
        params = urllib.parse.parse_qs(query_string)
        for k, v in params.items():
            if k not in fields and v:
                fields[k] = v[0]

        if not saved_file_path or not saved_file_path.is_file():
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No valid video file uploaded"}).encode("utf-8"))
            return

        stem_clean = Path(original_filename).stem if original_filename else "video"
        stem_clean = stem_clean.replace(" ", "_").replace("-", "_")

        scene_id = fields.get("scene_id", "").strip() or stem_clean
        session_id = fields.get("session_id", "").strip()
        if not session_id:
            t_str = time.strftime("%Y%m%d_%H%M%S")
            session_id = f"{scene_id}_{t_str}"

        point_stride = int(fields.get("point_stride", 2))
        frame_stride = int(fields.get("frame_stride", 1))
        voxel_size = float(fields.get("voxel_size", 0.003))
        confidence_threshold = float(fields.get("confidence_threshold", 0.1))
        auto_fuse = str(fields.get("auto_fuse", "true")).lower() in ("true", "1", "yes")
        dynamic_filter = str(fields.get("dynamic_filter", "false")).lower() in ("true", "1", "yes")
        wait_for_completion = str(fields.get("wait", "false")).lower() in ("true", "1", "yes")

        task_id = f"task_{session_id}"
        with _VIDEO_TASKS_LOCK:
            _VIDEO_TASKS[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0.0,
                "message": "视频已上传，排队等待 3D 建图推理...",
                "scene_id": scene_id,
                "session_id": session_id,
                "video_path": str(saved_file_path),
                "original_filename": original_filename,
                "point_stride": point_stride,
                "frame_stride": frame_stride,
                "voxel_size": voxel_size,
                "confidence_threshold": confidence_threshold,
                "auto_fuse": auto_fuse,
                "dynamic_filter": dynamic_filter,
                "created_at": time.time(),
                "updated_at": time.time(),
                "current_frame": 0,
                "total_frames": 0,
                "fps": 0.0,
                "result": None,
                "error": None,
            }

        worker = threading.Thread(
            target=run_video_pipeline_thread,
            kwargs={
                "task_id": task_id,
                "video_path": saved_file_path,
                "scene_id": scene_id,
                "session_id": session_id,
                "point_stride": point_stride,
                "frame_stride": frame_stride,
                "voxel_size": voxel_size,
                "confidence_threshold": confidence_threshold,
                "auto_fuse": auto_fuse,
                "dynamic_filter": dynamic_filter,
            },
            daemon=True,
        )
        worker.start()

        if wait_for_completion:
            while True:
                with _VIDEO_TASKS_LOCK:
                    t_info = _VIDEO_TASKS.get(task_id, {})
                    status = t_info.get("status")
                if status in ("completed", "failed"):
                    break
                time.sleep(0.5)

            self.send_response(200 if status == "completed" else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(t_info, ensure_ascii=False).encode("utf-8"))
            return

        resp_data = {
            "status": "queued",
            "task_id": task_id,
            "scene_id": scene_id,
            "session_id": session_id,
            "message": "视频已成功接收并加入建图任务队列",
            "status_url": f"/api/video/status?task_id={task_id}",
            "progress_url": f"/api/video/progress?task_id={task_id}",
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode("utf-8"))

    def handle_video_status(self, query_string: str) -> None:
        """Query status of a specific video reconstruction task or list all."""
        params = urllib.parse.parse_qs(query_string)
        task_id = params.get("task_id", [""])[0]
        with _VIDEO_TASKS_LOCK:
            if not task_id:
                tasks = list(_VIDEO_TASKS.values())
                data = {"tasks": tasks}
            elif task_id in _VIDEO_TASKS:
                data = _VIDEO_TASKS[task_id]
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Task '{task_id}' not found"}).encode("utf-8"))
                return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def handle_video_tasks(self) -> None:
        """Return list of recent video reconstruction tasks."""
        with _VIDEO_TASKS_LOCK:
            tasks = list(_VIDEO_TASKS.values())
        tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(tasks, ensure_ascii=False).encode("utf-8"))

    def handle_video_progress_sse(self, query_string: str) -> None:
        """Server-Sent Events stream for real-time task progress reporting."""
        params = urllib.parse.parse_qs(query_string)
        task_id = params.get("task_id", [""])[0]
        if not task_id:
            self.send_error(400, "Missing task_id parameter")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_progress = -1.0
        last_status = ""
        while True:
            with _VIDEO_TASKS_LOCK:
                task = _VIDEO_TASKS.get(task_id)
                if not task:
                    break
                status = task.get("status", "")
                progress = task.get("progress", 0.0)
                snapshot = dict(task)

            if progress != last_progress or status != last_status:
                last_progress = progress
                last_status = status
                try:
                    payload = f"event: progress\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break

            if status in ("completed", "failed"):
                try:
                    event_name = "complete" if status == "completed" else "error"
                    payload = f"event: {event_name}\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
                break

            time.sleep(0.3)

    def handle_scenes_list(self) -> None:
        """List all registered scenes in outputs/scenes/."""
        scenes_dir = ROOT_DIR / "outputs" / "scenes"
        results = []
        if scenes_dir.is_dir():
            for d in sorted(scenes_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    ply = d / "reconstruction.ply"
                    colored_ply = d / "reconstruction_colored.ply"
                    meta_file = d / "scene_metadata.json"
                    meta = {}
                    if meta_file.is_file():
                        try:
                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    results.append({
                        "scene_id": d.name,
                        "has_ply": ply.is_file(),
                        "has_colored_ply": colored_ply.is_file(),
                        "ply_size_mb": round(ply.stat().st_size / 1024 / 1024, 2) if ply.is_file() else 0.0,
                        "streams": meta.get("completed_sessions", []),
                        "is_fused": meta.get("is_fused", False),
                        "updated_at": meta.get("updated_at"),
                    })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(results, ensure_ascii=False).encode("utf-8"))


    def handle_8090_status(self) -> None:
        """Check whether 8090 streaming engine is listening and operational."""
        alive = is_port_listening(8090)
        info = {"running": alive, "port": 8090, "ready": alive}
        if alive:
            try:
                req = urllib.request.Request("http://127.0.0.1:8090/", headers={"User-Agent": "ABot-8088"})
                with _NO_PROXY_OPENER.open(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        info["ready"] = True
            except Exception:
                info["ready"] = True  # Port is actively listening and alive
        else:
            info["ready"] = False

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(info).encode("utf-8"))

    def handle_8090_sessions(self) -> None:
        """Query 8090 for currently active and completed streaming sessions."""
        global _LAST_8090_SESSIONS
        alive = is_port_listening(8090)
        res = {"status": "running" if alive else "stopped", "active_sessions": [], "completed_sessions": []}
        if alive:
            try:
                req = urllib.request.Request("http://127.0.0.1:8090/api/sessions", headers={"User-Agent": "ABot-8088"})
                with _NO_PROXY_OPENER.open(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        res = {
                            "status": "running",
                            "active_sessions": data.get("active_sessions", []),
                            "completed_sessions": data.get("completed_sessions", []),
                        }
                        _LAST_8090_SESSIONS = res
            except Exception:
                # If high-load GPU processing caused a temporary delay, preserve last known active sessions so UI never flickers
                if _LAST_8090_SESSIONS and _LAST_8090_SESSIONS.get("status") == "running":
                    res = _LAST_8090_SESSIONS
        else:
            _LAST_8090_SESSIONS = {"status": "stopped", "active_sessions": [], "completed_sessions": []}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))

    def handle_8090_live_sse(self, query_string: str) -> None:
        """Relay real-time 8090 SSE events to the browser client on 8088."""
        params = urllib.parse.parse_qs(query_string)
        session_id = params.get("session_id", [""])[0]

        target_url = "http://127.0.0.1:8090/api/stream/live"
        if session_id:
            target_url += f"/{urllib.parse.quote(session_id)}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            req = urllib.request.Request(target_url, headers={"Accept": "text/event-stream", "User-Agent": "ABot-8088"})
            with _NO_PROXY_OPENER.open(req, timeout=3600) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    if line == b"\n":
                        self.wfile.flush()
        except Exception as e:
            try:
                err_data = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                self.wfile.write(err_data.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def handle_8090_snapshot(self, query_string: str) -> None:
        """Relay session 3D reconstruction snapshot from 8090."""
        params = urllib.parse.parse_qs(query_string)
        session_id = params.get("session_id", [""])[0]
        if not session_id:
            self.send_error(400, "Missing session_id parameter")
            return
        target_url = f"http://127.0.0.1:8090/api/stream/snapshot/{urllib.parse.quote(session_id)}"
        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "ABot-8088"})
            with _NO_PROXY_OPENER.open(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Failed to fetch snapshot from 8090: {e}")

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


def run_server(port: int = 8088, host: str = "0.0.0.0", device: str = "cuda") -> None:
    server_address = (host, port)
    ThreadingHTTPServer.allow_reuse_address = True

    # Pre-warm model and compile JIT kernels on GPU
    engine = get_engine(device=device)
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
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run on (default: cuda:0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host, device=args.device)

if __name__ == "__main__":
    main()
