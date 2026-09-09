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
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from viewer.stream_backend import OnlineReconstructionEngine

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

        if path == "/api/sequences":
            return self.handle_sequences()
        if path == "/api/offline_models":
            return self.handle_offline_models()

        if path == "/api/status":
            return self.handle_status()
        if path == "/api/stop":
            return self.handle_stop_stream()
        return super().do_GET()

    def handle_offline_models(self) -> None:
        """List all reconstructed 3D point cloud models stored on disk."""
        models = []
        registry = [
            {
                "path": "outputs/alignment/data_05_08_method2_merged.ply",
                "name": "🔥 [方案二 多模态] 视频流05-08 正常真彩融合 (341万点, 1.5cm体素去重, 推荐)",
                "category": "Method 2 Multimodal 05-08",
                "description": "基于 ALIKED+LightGlue+Umeyama+small_gicp 对视频05-08多视角流式融合 (3,409,327 点，87.8MB)"
            },
            {
                "path": "outputs/alignment/data_05_08_method2_colored_merged.ply",
                "name": "🎨 [方案二 区分色彩] 视频流05-08 四色区分融合 (341万点, 红/绿/蓝/金, 推荐)",
                "category": "Method 2 Colored 05-08",
                "description": "每个视频流赋予独立高对比颜色（05红/06绿/07蓝/08金），直观清晰展现各视频点云空间分布"
            },
            {
                "path": "outputs/alignment/data_05_08_method2_full_merged.ply",
                "name": "🌟 [方案二 多模态] 视频流05-08 正常真彩全量融合 (1401万点超高清点云, 零损失)",
                "category": "Method 2 Multimodal 05-08",
                "description": "保留05-08全部 14,014,980 点，无损拼接完整全景走廊与开阔大厅 (361MB)"
            },
            {
                "path": "outputs/alignment/data_05_08_method2_colored_full_merged.ply",
                "name": "🎨 [方案二 区分色彩] 视频流05-08 四色区分全量融合 (1401万点超高清点云)",
                "category": "Method 2 Colored 05-08",
                "description": "1401万点全量四色点云（05红/06绿/07蓝/08金）"
            },
            {
                "path": "outputs/data_05_loop/reconstruction.ply",
                "name": "📹 视频流 05 - 点云重建 (567万点, A-B-C-B-A 循环全景)",
                "category": "05-08 Individual Videos",
                "description": "data/data/05 走廊循环视频流 (643 帧)，567万点超高清点云"
            },
            {
                "path": "outputs/data_06_loop/reconstruction.ply",
                "name": "📹 视频流 06 - 点云重建 (271万点, C-D 右侧视角)",
                "category": "05-08 Individual Videos",
                "description": "data/data/06 视频流 (307 帧)，271万点点云"
            },
            {
                "path": "outputs/data_07_loop/reconstruction.ply",
                "name": "📹 视频流 07 - 点云重建 (344万点, B-D 主干基准视角)",
                "category": "05-08 Individual Videos",
                "description": "data/data/07 视频流 (390 帧)，344万点点云 (多视角基准锚点)"
            },
            {
                "path": "outputs/data_08_loop/reconstruction.ply",
                "name": "📹 视频流 08 - 点云重建 (220万点, C-D 左侧视角)",
                "category": "05-08 Individual Videos",
                "description": "data/data/08 视频流 (249 帧)，220万点点云"
            },
            {
                "path": "outputs/alignment/method2_lightglue_umeyama_full_merged.ply",
                "name": "🔥 [方案二 多模态] 走廊实测 视频1+视频2 全量无损拼接 (LightGlue+Umeyama, 539万点, 推荐)",
                "category": "Method 2 Multimodal",
                "description": "基于 ALIKED+LightGlue 视频特征匹配与 Umeyama 求解相似变换融合 (5,389,020 点，78.87% @ 5cm)"
            },
            {
                "path": "outputs/alignment/method2_lightglue_umeyama_merged.ply",
                "name": "🌟 [方案二 多模态] 走廊实测 视频1+视频2 1.5cm去重融合 (49.3万点)",
                "category": "Method 2 Multimodal",
                "description": "1.5cm 体素去重精简版本，适合低配显卡极致流畅交互 (493,129 点，13MB)"
            },
            {
                "path": "outputs/alignment/method1_kiss_gicp_full_merged.ply",
                "name": "📐 [方案一 纯几何] 走廊实测 视频1+视频2 全量无损拼接 (KISS-Matcher+GICP, 539万点)",
                "category": "Method 1 Geometric",
                "description": "纯 3D 几何特征 Faster-PFH + small_gicp 并行对齐无损拼接 (5,389,020 点，75.85% @ 5cm)"
            },
            {
                "path": "outputs/alignment/method1_kiss_gicp_merged.ply",
                "name": "📐 [方案一 纯几何] 走廊实测 视频1+视频2 1.5cm去重融合 (46.4万点)",
                "category": "Method 1 Geometric",
                "description": "1.5cm 体素去重平滑过渡版本 (463,701 点，12MB)"
            },
            {
                "path": "outputs/alignment/merged.ply",
                "name": "🤖 [方案三 深度学习] 走廊实测 视频1+视频2 R3PM-Net全量融合 (merged.ply, 539万点)",
                "category": "R3PM-Net Merged",
                "description": "基于 R3PM-Net 深度点匹配网络与 Sinkhorn 对应估计对齐 (5,389,020 点，80.8MB)"
            },
            {
                "path": "outputs/alignment/mine_r3pm_net_5mm_merged.ply",
                "name": "🤖 [方案三 深度学习] 走廊实测 视频1+视频2 5mm去重融合 (276万点)",
                "category": "R3PM-Net Merged",
                "description": "5mm 接触面体素去重平滑过渡版本 (2,759,565 点，39.5MB)"
            },
            {
                "path": "outputs/mine_VID20260903181931_loop/reconstruction.ply",
                "name": "📹 自定义视频 1 - 回环优化 (VID181931, 279万点, 走廊实测)",
                "category": "My Videos",
                "description": "data/mine 走廊实测视频流 (316 帧)，279万点超高清点云"
            },
            {
                "path": "outputs/mine_VID20260903181931_noloop/reconstruction.ply",
                "name": "📹 自定义视频 1 - 原始流式 (VID181931, 279万点)",
                "category": "My Videos",
                "description": "data/mine 走廊视频 1，纯因果流式预测"
            },
            {
                "path": "outputs/mine_VID20260903182041_loop/reconstruction.ply",
                "name": "📹 自定义视频 2 - 回环优化 (VID182041, 260万点, 走廊实测)",
                "category": "My Videos",
                "description": "data/mine 走廊实测视频流 (295 帧)，260万点超高清点云"
            },
            {
                "path": "outputs/mine_VID20260903182041_noloop/reconstruction.ply",
                "name": "📹 自定义视频 2 - 原始流式 (VID182041, 260万点)",
                "category": "My Videos",
                "description": "data/mine 走廊视频 2，纯因果流式预测"
            },
            {
                "path": "outputs/alignment/tum_method2_lightglue_umeyama_full_merged.ply",
                "name": "🔥 [TUM 方案二] 360+Desk 100%全量无损拼接 (604万超高清点云, 推荐)",
                "category": "TUM Merged",
                "description": "保留全部 333万+270万 原始点云，零点数损失 (6,041,700 点，86MB)"
            },
            {
                "path": "outputs/alignment/tum_method1_kiss_gicp_full_merged.ply",
                "name": "🔥 [TUM 方案一] 360+Desk 100%全量无损拼接 (604万超高清点云)",
                "category": "TUM Merged",
                "description": "纯几何配准全量拼接，零点数损失 (6,041,700 点，86MB)"
            },
            {
                "path": "outputs/alignment/tum_method2_lightglue_umeyama_merged.ply",
                "name": "🌟 [TUM 方案二] 360+Desk 5mm去重融合 (LightGlue+Umeyama, 222万点)",
                "category": "TUM Merged",
                "description": "5mm 接触面体素去重平滑过渡版本 (2,217,810 点，31MB)"
            },
            {
                "path": "outputs/alignment/tum_method1_kiss_gicp_merged.ply",
                "name": "📐 [TUM 方案一] 360+Desk 5mm去重融合 (KISS-Matcher+GICP, 226万点)",
                "category": "TUM Merged",
                "description": "5mm 接触面体素去重平滑过渡版本 (2,261,226 点，32MB)"
            },
            {
                "path": "outputs/tum_360_loop/reconstruction.ply",
                "name": "🔄 TUM 360环绕 - 回环优化 (Loop Closure, 333万点)",
                "category": "TUM 360",
                "description": "360度大环绕轨迹对齐，消除闭环双层重影"
            },
            {
                "path": "outputs/tum_360_noloop/reconstruction.ply",
                "name": "🔄 TUM 360环绕 - 原始流式 (No Loop, 333万点)",
                "category": "TUM 360",
                "description": "纯因果单向累加，观察长程旋转下的轨迹与几何漂移"
            },
            {
                "path": "outputs/tum_desk_loop/reconstruction.ply",
                "name": "🖥️ TUM 办公桌面 - 回环优化 (Loop Closure, 270万点)",
                "category": "TUM Desk",
                "description": "电脑显示器/书籍/键盘，回环位姿图平滑对齐"
            },
            {
                "path": "outputs/tum_desk_noloop/reconstruction.ply",
                "name": "🖥️ TUM 办公桌面 - 原始流式 (No Loop, 270万点)",
                "category": "TUM Desk",
                "description": "纯因果单向流式累加"
            },
            {
                "path": "outputs/demo_loop/reconstruction.ply",
                "name": "🎬 快速演示序列 - 回环优化 (52.9万点)",
                "category": "Demo",
                "description": "60 帧快速测试序列"
            },
            {
                "path": "outputs/demo_noloop/reconstruction.ply",
                "name": "🚀 快速演示序列 - 原始流式 (52.9万点)",
                "category": "Demo",
                "description": "60 帧快速测试序列"
            },
        ]

        for item in registry:
            ply_file = ROOT_DIR / item["path"]
            if ply_file.is_file():
                models.append({
                    "url": f"/{item['path']}",
                    "name": item["name"],
                    "category": item["category"],
                    "description": item["description"],
                    "size_bytes": ply_file.stat().st_size,
                })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(models, ensure_ascii=False).encode("utf-8"))
        return

    def handle_sequences(self) -> None:
        """Scan and list all available video datasets and image sequences."""
        sequences = []

        registry = [
            {
                "id": "data/data/05",
                "name": "📹 视频流 05 (A-B-C-B-A 循环全景)",
                "path": "data/data/05",
                "description": "A狭窄走廊出发走至B右转90°到C，绕书柜转180°回B左转直走回A (643 帧)"
            },
            {
                "id": "data/data/06",
                "name": "📹 视频流 06 (C-D 右侧视角)",
                "path": "data/data/06",
                "description": "从目标区域右侧走过 (镜头朝左) C点到D点 (307 帧)"
            },
            {
                "id": "data/data/07",
                "name": "📹 视频流 07 (B-D 主干基准视角)",
                "path": "data/data/07",
                "description": "与06同路线，摄像头运动基本相同，B点到D点全程 (390 帧)"
            },
            {
                "id": "data/data/08",
                "name": "📹 视频流 08 (C-D 左侧视角)",
                "path": "data/data/08",
                "description": "与06同区域，但从左侧走过 (镜头朝右) C点到D点 (249 帧)"
            },
            {
                "id": "data/mine/VID20260903181931",
                "name": "📹 用户实拍视频 1 (走廊流式 VID181931)",
                "path": "data/mine/VID20260903181931",
                "description": "data/mine 实拍走廊视频 (316 帧)"
            },
            {
                "id": "data/mine/VID20260903182041",
                "name": "📹 用户实拍视频 2 (走廊流式 VID182041)",
                "path": "data/mine/VID20260903182041",
                "description": "data/mine 实拍走廊视频 (295 帧)"
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_desk/rgb",
                "name": "🖥️ TUM 办公桌面全景 (Desk Sequence)",
                "path": "data/tum/rgbd_dataset_freiburg1_desk/rgb",
                "description": "办公桌全景、电脑显示器、键盘、书籍 (613 帧)"
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_xyz/rgb",
                "name": "📐 TUM 空间平移序列 (XYZ Motion)",
                "path": "data/tum/rgbd_dataset_freiburg1_xyz/rgb",
                "description": "沿 X/Y/Z 三轴典型平移扫描 (798 帧)"
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_360/rgb",
                "name": "🔄 TUM 360度环绕回环 (360 Loop)",
                "path": "data/tum/rgbd_dataset_freiburg1_360/rgb",
                "description": "绕桌面 360 度环绕拍摄，经典回环场景 (756 帧)"
            },
            {
                "id": "data/tum/rgbd_dataset_freiburg1_room/rgb",
                "name": "🏢 TUM 完整大房间场景 (Full Room)",
                "path": "data/tum/rgbd_dataset_freiburg1_room/rgb",
                "description": "完整办公室大场景、多张桌椅、黑板 (1362 帧)"
            },
            {
                "id": "examples/images",
                "name": "🎬 快速演示序列 (Demo Sample)",
                "path": "examples/images",
                "description": "TUM 办公桌局部平移 (60 帧快速体验)"
            },
        ]

        for item in registry:
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
                        "frames": frames,
                        "description": item["description"],
                    })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(sequences, ensure_ascii=False).encode("utf-8"))
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

        image_dir = ROOT_DIR / image_dir_name
        if not image_dir.is_dir():
            self.send_error(404, f"Image directory {image_dir_name} not found")
            return

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
    def log_message(self, format: str, *args) -> None:
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
