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


# Curated models registry for rich metadata preservation
CURATED_MODELS_REGISTRY: Dict[str, Dict[str, str]] = {
    "outputs/alignment/perfect_corridor_merged.ply": {
        "name": "🏆 [走廊直道对齐 最佳] 测试走廊 视频1+视频2 统一轴线融合 (23.3万点, 15mm真彩, 推荐)",
        "category": "走廊精准直道融合 (Perfect Corridor Fusion)",
        "description": "消除两端反向录制坐标系偏航角差异，93.9%重叠率精确对齐同一个书架与显示器 (233,115 点，6.0MB)",
    },
    "outputs/alignment/perfect_corridor_colored_merged.ply": {
        "name": "🎨 [走廊直道对齐 双色检验] 视频1(红) + 视频2(绿) 统一轴线标定 (推荐)",
        "category": "走廊精准直道融合 (Perfect Corridor Fusion)",
        "description": "视频1红色、视频2绿色高对比标定，直观检验书架、地面与显示器空间 100% 重叠吻合",
    },
    "outputs/alignment/data_05_08_method2_merged.ply": {
        "name": "🔥 [方案二 多模态] 视频流05-08 正常真彩融合 (341万点, 1.5cm体素去重, 推荐)",
        "category": "Method 2 Multimodal 05-08",
        "description": "基于 ALIKED+LightGlue+Umeyama+small_gicp 对视频05-08多视角流式融合 (3,409,327 点，87.8MB)",
    },
    "outputs/alignment/data_05_08_method2_colored_merged.ply": {
        "name": "🎨 [方案二 区分色彩] 视频流05-08 四色区分融合 (341万点, 红/绿/蓝/金, 推荐)",
        "category": "Method 2 Colored 05-08",
        "description": "每个视频流赋予独立高对比颜色（05红/06绿/07蓝/08金），直观清晰展现各视频点云空间分布",
    },
    "outputs/alignment/data_05_08_method2_full_merged.ply": {
        "name": "🌟 [方案二 多模态] 视频流05-08 正常真彩全量融合 (1401万点超高清点云, 零损失)",
        "category": "Method 2 Multimodal 05-08",
        "description": "保留05-08全部 14,014,980 点，无损拼接完整全景走廊与开阔大厅 (361MB)",
    },
    "outputs/alignment/data_05_08_method2_colored_full_merged.ply": {
        "name": "🎨 [方案二 区分色彩] 视频流05-08 四色区分全量融合 (1401万点超高清点云)",
        "category": "Method 2 Colored 05-08",
        "description": "1401万点全量四色点云（05红/06绿/07蓝/08金）",
    },
    "outputs/data_05_loop/reconstruction.ply": {
        "name": "📹 视频流 05 - 点云重建 (567万点, A-B-C-B-A 循环全景)",
        "category": "05-08 Individual Videos",
        "description": "data/data/05 走廊循环视频流 (643 帧)，567万点超高清点云",
    },
    "outputs/data_06_loop/reconstruction.ply": {
        "name": "📹 视频流 06 - 点云重建 (271万点, C-D 右侧视角)",
        "category": "05-08 Individual Videos",
        "description": "data/data/06 视频流 (307 帧)，271万点点云",
    },
    "outputs/data_07_loop/reconstruction.ply": {
        "name": "📹 视频流 07 - 点云重建 (344万点, B-D 主干基准视角)",
        "category": "05-08 Individual Videos",
        "description": "data/data/07 视频流 (390 帧)，344万点点云 (多视角基准锚点)",
    },
    "outputs/data_08_loop/reconstruction.ply": {
        "name": "📹 视频流 08 - 点云重建 (220万点, C-D 左侧视角)",
        "category": "05-08 Individual Videos",
        "description": "data/data/08 视频流 (249 帧)，220万点点云",
    },
    "outputs/alignment/method2_lightglue_umeyama_full_merged.ply": {
        "name": "🔥 [方案二 多模态] 走廊实测 视频1+视频2 全量无损拼接 (LightGlue+Umeyama, 539万点, 推荐)",
        "category": "Method 2 Multimodal",
        "description": "基于 ALIKED+LightGlue 视频特征匹配与 Umeyama 求解相似变换融合 (5,389,020 点，78.87% @ 5cm)",
    },
    "outputs/alignment/method2_lightglue_umeyama_merged.ply": {
        "name": "🌟 [方案二 多模态] 走廊实测 视频1+视频2 1.5cm去重融合 (49.3万点)",
        "category": "Method 2 Multimodal",
        "description": "1.5cm 体素去重精简版本，适合低配显卡极致流畅交互 (493,129 点，13MB)",
    },
    "outputs/alignment/method1_kiss_gicp_full_merged.ply": {
        "name": "📐 [方案一 纯几何] 走廊实测 视频1+视频2 全量无损拼接 (KISS-Matcher+GICP, 539万点)",
        "category": "Method 1 Geometric",
        "description": "纯 3D 几何特征 Faster-PFH + small_gicp 并行对齐无损拼接 (5,389,020 点，75.85% @ 5cm)",
    },
    "outputs/alignment/method1_kiss_gicp_merged.ply": {
        "name": "📐 [方案一 纯几何] 走廊实测 视频1+视频2 1.5cm去重融合 (46.4万点)",
        "category": "Method 1 Geometric",
        "description": "1.5cm 体素去重平滑过渡版本 (463,701 点，12MB)",
    },
    "outputs/alignment/merged.ply": {
        "name": "🤖 [方案三 深度学习] 走廊实测 视频1+视频2 R3PM-Net全量融合 (merged.ply, 539万点)",
        "category": "R3PM-Net Merged",
        "description": "基于 R3PM-Net 深度点匹配网络与 Sinkhorn 对应估计对齐 (5,389,020 点，80.8MB)",
    },
    "outputs/alignment/mine_r3pm_net_5mm_merged.ply": {
        "name": "🤖 [方案三 深度学习] 走廊实测 视频1+视频2 5mm去重融合 (276万点)",
        "category": "R3PM-Net Merged",
        "description": "5mm 接触面体素去重平滑过渡版本 (2,759,565 点，39.5MB)",
    },
    "outputs/mine_VID20260903181931_loop/reconstruction.ply": {
        "name": "📹 自定义视频 1 - 回环优化 (VID181931, 279万点, 走廊实测)",
        "category": "My Videos",
        "description": "data/mine 走廊实测视频流 (316 帧)，279万点超高清点云",
    },
    "outputs/mine_VID20260903181931_noloop/reconstruction.ply": {
        "name": "📹 自定义视频 1 - 原始流式 (VID181931, 279万点)",
        "category": "My Videos",
        "description": "data/mine 走廊视频 1，纯因果流式预测",
    },
    "outputs/mine_VID20260903182041_loop/reconstruction.ply": {
        "name": "📹 自定义视频 2 - 回环优化 (VID182041, 260万点, 走廊实测)",
        "category": "My Videos",
        "description": "data/mine 走廊实测视频流 (295 帧)，260万点超高清点云",
    },
    "outputs/mine_VID20260903182041_noloop/reconstruction.ply": {
        "name": "📹 自定义视频 2 - 原始流式 (VID182041, 260万点)",
        "category": "My Videos",
        "description": "data/mine 走廊视频 2，纯因果流式预测",
    },
    "outputs/alignment/tum_method2_lightglue_umeyama_full_merged.ply": {
        "name": "🔥 [TUM 方案二] 360+Desk 100%全量无损拼接 (604万超高清点云, 推荐)",
        "category": "TUM Merged",
        "description": "保留全部 333万+270万 原始点云，零点数损失 (6,041,700 点，86MB)",
    },
    "outputs/alignment/tum_method1_kiss_gicp_full_merged.ply": {
        "name": "🔥 [TUM 方案一] 360+Desk 100%全量无损拼接 (604万超高清点云)",
        "category": "TUM Merged",
        "description": "纯几何配准全量拼接，零点数损失 (6,041,700 点，86MB)",
    },
    "outputs/alignment/tum_method2_lightglue_umeyama_merged.ply": {
        "name": "🌟 [TUM 方案二] 360+Desk 5mm去重融合 (LightGlue+Umeyama, 222万点)",
        "category": "TUM Merged",
        "description": "5mm 接触面体素去重平滑过渡版本 (2,217,810 点，31MB)",
    },
    "outputs/alignment/tum_method1_kiss_gicp_merged.ply": {
        "name": "📐 [TUM 方案一] 360+Desk 5mm去重融合 (KISS-Matcher+GICP, 226万点)",
        "category": "TUM Merged",
        "description": "5mm 接触面体素去重平滑过渡版本 (2,261,226 点，32MB)",
    },
    "outputs/tum_360_loop/reconstruction.ply": {
        "name": "🔄 TUM 360环绕 - 回环优化 (Loop Closure, 333万点)",
        "category": "TUM 360",
        "description": "360度大环绕轨迹对齐，消除闭环双层重影",
    },
    "outputs/tum_360_noloop/reconstruction.ply": {
        "name": "🔄 TUM 360环绕 - 原始流式 (No Loop, 333万点)",
        "category": "TUM 360",
        "description": "纯因果单向累加，观察长程旋转下的轨迹与几何漂移",
    },
    "outputs/tum_desk_loop/reconstruction.ply": {
        "name": "🖥️ TUM 办公桌面 - 回环优化 (Loop Closure, 270万点)",
        "category": "TUM Desk",
        "description": "电脑显示器/书籍/键盘，回环位姿图平滑对齐",
    },
    "outputs/tum_desk_noloop/reconstruction.ply": {
        "name": "🖥️ TUM 办公桌面 - 原始流式 (No Loop, 270万点)",
        "category": "TUM Desk",
        "description": "纯因果单向流式累加",
    },
    "outputs/demo_loop/reconstruction.ply": {
        "name": "🎬 快速演示序列 - 回环优化 (52.9万点)",
        "category": "Demo",
        "description": "60 帧快速测试序列",
    },
    "outputs/demo_noloop/reconstruction.ply": {
        "name": "🚀 快速演示序列 - 原始流式 (52.9万点)",
        "category": "Demo",
        "description": "60 帧快速测试序列",
    },
}

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

def scan_all_ply_models() -> List[Dict[str, Any]]:
    """Scan /home/data/xyz/ABot-Recon/outputs recursively for all .ply point clouds."""
    outputs_dir = ROOT_DIR / "outputs"
    if not outputs_dir.is_dir():
        return []

    ply_files = sorted(outputs_dir.glob("**/*.ply"))
    models = []

    category_order = {
        "走廊精准直道融合 (Perfect Corridor Fusion)": -2,
        "Method 2 Multimodal 05-08": 0,
        "Method 2 Colored 05-08": 1,
        "多视频流场景融合点云 (Multi-Stream Scenes)": 2,
        "通用多视角融合 (General Fusion)": 3,
        "Method 2 Multimodal": 4,
        "Method 1 Geometric": 4,
        "R3PM-Net Merged": 5,
        "TUM Merged": 6,
        "05-08 Individual Videos": 7,
        "单视频流点云 (Single Video)": 8,
        "动态滤波对比 (Dynamic Filtering)": 9,
        "My Videos": 10,
        "TUM 360": 11,
        "TUM Desk": 12,
        "Demo": 13,
        "实时流式会话点云 (Streaming Sessions)": 14,
        "回环对比点云 (Loop Comparison)": 15,
        "配准与多路融合 (Alignment & Fusion)": 16,
        "Outputs 其他点云 (Other Models)": 17,
    }

    for ply_path in ply_files:
        rel_str = str(ply_path.relative_to(ROOT_DIR))
        url = f"/{rel_str}"
        vertex_count, size_mb = get_ply_header_info(ply_path)
        size_bytes = ply_path.stat().st_size

        if rel_str in CURATED_MODELS_REGISTRY:
            curated = CURATED_MODELS_REGISTRY[rel_str]
            name = curated["name"]
            category = curated["category"]
            description = curated["description"]
        else:
            if rel_str.startswith("outputs/scenes/"):
                scene_dir = ply_path.parent
                scene_name = scene_dir.name
                category = "多视频流场景融合点云 (Multi-Stream Scenes)"

                # Deduplicate: if reconstruction.ply / reconstruction_colored.ply exists,
                # skip redundant *_normal_merged.ply / *_colored_merged.ply copies
                if ply_path.name.endswith("_normal_merged.ply") and (scene_dir / "reconstruction.ply").is_file():
                    continue
                if ply_path.name.endswith("_colored_merged.ply") and (scene_dir / "reconstruction_colored.ply").is_file():
                    continue

                is_colored = "colored" in ply_path.name
                is_full = "full" in ply_path.name
                if is_colored:
                    tag = "🎨 [场景区分色彩-全量]" if is_full else "🎨 [场景区分色彩]"
                else:
                    tag = "🌟 [场景全景融合-全量]" if is_full else "🏛️ [场景全景融合]"
                name = f"{tag} {scene_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            elif rel_str.startswith("outputs/alignment/general_fusion/"):
                category = "通用多视角融合 (General Fusion)"
                stem = ply_path.stem
                is_colored = "colored" in stem
                is_full = "full" in stem
                tag = "🎨 [多模态区分色彩]" if is_colored else "🔥 [多模态正常真彩]"
                mode_str = "全量无损" if is_full else "体素去重"
                name = f"{tag} {stem} ({mode_str}, {format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            elif rel_str.startswith("outputs/alignment/"):
                category = "配准与多路融合 (Alignment & Fusion)"
                name = f"📐 [配准融合] {ply_path.stem} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            elif rel_str.startswith("outputs/streams/"):
                session_name = ply_path.parent.name
                category = "实时流式会话点云 (Streaming Sessions)"
                name = f"📡 [流式会话] {session_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            elif rel_str.startswith("outputs/loop_comparison/"):
                category = "回环对比点云 (Loop Comparison)"
                name = f"🔄 [回环对比] {ply_path.stem} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            elif ply_path.name.startswith("reconstruction"):
                seq_dir = ply_path.parent
                seq_name = seq_dir.name
                fname = ply_path.name

                # If reconstruction.ply and reconstruction_scheme1_filtered.ply both exist, skip redundant copy
                if fname == "reconstruction_scheme1_filtered.ply" and (seq_dir / "reconstruction.ply").is_file():
                    continue

                if fname == "reconstruction_clean.ply":
                    category = "动态滤波对比 (Dynamic Filtering)"
                    name = f"🧹 [滤波去噪] {seq_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
                elif fname == "reconstruction_baseline_with_dynamic.ply":
                    category = "动态滤波对比 (Dynamic Filtering)"
                    name = f"🚶 [原始含动态基线] {seq_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
                elif fname == "reconstruction_removed_dynamic_only.ply":
                    category = "动态滤波对比 (Dynamic Filtering)"
                    name = f"📦 [仅动态物体] {seq_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
                elif fname == "reconstruction_scheme1_filtered.ply":
                    category = "动态滤波对比 (Dynamic Filtering)"
                    name = f"🛡️ [动态过滤] {seq_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
                else:
                    category = "单视频流点云 (Single Video)"
                    name = f"📹 [单视频重建] {seq_name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"
            else:
                category = "Outputs 其他点云 (Other Models)"
                name = f"📁 {ply_path.parent.name}/{ply_path.name} ({format_point_count(vertex_count)}, {size_mb:.1f}MB)"

            description = f"文件路径: {rel_str} | 大小: {size_mb:.2f}MB" + (f" | 点数: {vertex_count:,}" if vertex_count else "")

        models.append({
            "url": url,
            "path": rel_str,
            "name": name,
            "category": category,
            "description": description,
            "size_bytes": size_bytes,
            "vertex_count": vertex_count,
            "_order": category_order.get(category, 50),
        })
    def get_sort_key(m):
        prio = 1
        fname = Path(m["path"]).name
        if fname in ("data_05_08_method2_merged.ply", "data_05_08_method2_colored_merged.ply", "reconstruction.ply"):
            prio = 0
        return (m["_order"], prio, m["name"])

    models.sort(key=get_sort_key)
    for m in models:
        m.pop("_order", None)
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

        # 2. Raw video sequences
        video_registry = [
            {
                "id": "data/data/05",
                "name": "📹 视频流 05 (A-B-C-B-A 循环全景)",
                "path": "data/data/05",
                "description": "A狭窄走廊出发走至B右转90°到C，绕书柜转180°回B左转直走回A (643 帧)",
            },
            {
                "id": "data/data/06",
                "name": "📹 视频流 06 (C-D 右侧视角)",
                "path": "data/data/06",
                "description": "从目标区域右侧走过 (镜头朝左) C点到D点 (307 帧)",
            },
            {
                "id": "data/data/07",
                "name": "📹 视频流 07 (B-D 主干基准视角)",
                "path": "data/data/07",
                "description": "与06同路线，摄像头运动基本相同，B点到D点全程 (390 帧)",
            },
            {
                "id": "data/data/08",
                "name": "📹 视频流 08 (C-D 左侧视角)",
                "path": "data/data/08",
                "description": "与06同区域，但从左侧走过 (镜头朝右) C点到D点 (249 帧)",
            },
            {
                "id": "data/mine/VID20260903181931",
                "name": "📹 用户实拍视频 1 (走廊流式 VID181931)",
                "path": "data/mine/VID20260903181931",
                "description": "data/mine 实拍走廊视频 (316 帧)",
            },
            {
                "id": "data/mine/VID20260903182041",
                "name": "📹 用户实拍视频 2 (走廊流式 VID182041)",
                "path": "data/mine/VID20260903182041",
                "description": "data/mine 实拍走廊视频 (295 帧)",
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_desk/rgb",
                "name": "🖥️ TUM 办公桌面全景 (Desk Sequence)",
                "path": "data/tum/rgbd_dataset_freiburg1_desk/rgb",
                "description": "办公桌全景、电脑显示器、键盘、书籍 (613 帧)",
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_xyz/rgb",
                "name": "📐 TUM 空间平移序列 (XYZ Motion)",
                "path": "data/tum/rgbd_dataset_freiburg1_xyz/rgb",
                "description": "沿 X/Y/Z 三轴典型平移扫描 (798 帧)",
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_360/rgb",
                "name": "🔄 TUM 360度环绕回环 (360 Loop)",
                "path": "data/tum/rgbd_dataset_freiburg1_360/rgb",
                "description": "绕桌面 360 度环绕拍摄，经典回环场景 (756 帧)",
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_room/rgb",
                "name": "🏢 TUM 完整大房间场景 (Full Room)",
                "path": "data/tum/rgbd_dataset_freiburg1_room/rgb",
                "description": "完整办公室大场景、多张桌椅、黑板 (1362 帧)",
            },
            {
                "id": "examples/images",
                "name": "🎬 快速演示序列 (Demo Sample)",
                "path": "examples/images",
                "description": "TUM 办公桌局部平移 (60 帧快速体验)",
            },
        ]

        for item in video_registry:
            seq_dir = ROOT_DIR / item["path"]
            if seq_dir.is_dir():
                frames = len([
                    p for p in seq_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                ])
                if frames > 0:
                    sequences.append({
                        "id": item["id"],
                        "name": f"{item['name']} - {frames} 帧",
                        "path": item["path"],
                        "category": "📹 原始视频流序列 (GPU 深度在线推理建图)",
                        "type": "video",
                        "frames": frames,
                        "description": item["description"],
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
