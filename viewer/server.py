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

        return super().do_GET()

    def handle_offline_models(self) -> None:
        """List all reconstructed 3D point cloud models stored on disk."""
        models = []
        registry = [
            {
                "path": "outputs/mine_VID20260903153130_loop/reconstruction.ply",
                "name": "📹 自定义视频 3 - 回环优化 (VID153130, 310万点)",
                "category": "My Videos",
                "description": "data/mine 第3段视频 (352 帧)，310万点超高清点云"
            },
            {
                "path": "outputs/mine_VID20260903153130_noloop/reconstruction.ply",
                "name": "📹 自定义视频 3 - 原始流式 (VID153130, 310万点)",
                "category": "My Videos",
                "description": "data/mine 第3段视频，纯因果流式预测"
            },
            {
                "path": "outputs/mine_VID20260903153215_loop/reconstruction.ply",
                "name": "📹 自定义视频 4 - 回环优化 (VID153215, 243万点)",
                "category": "My Videos",
                "description": "data/mine 第4段视频 (275 帧)，243万点超高清点云"
            },
            {
                "path": "outputs/mine_VID20260903153215_noloop/reconstruction.ply",
                "name": "📹 自定义视频 4 - 原始流式 (VID153215, 243万点)",
                "category": "My Videos",
                "description": "data/mine 第4段视频，纯因果流式预测"
            },
            {
                "path": "outputs/mine_VID20260903151228_loop/reconstruction.ply",
                "name": "📹 自定义视频 2 - 回环优化 (VID151228, 263万点)",
                "category": "My Videos",
                "description": "data/mine 第2段视频 (298 帧)，263万点三维致密点云"
            },
            {
                "path": "outputs/mine_VID20260903151228_noloop/reconstruction.ply",
                "name": "📹 自定义视频 2 - 原始流式 (VID151228, 263万点)",
                "category": "My Videos",
                "description": "data/mine 第2段视频，纯因果流式预测"
            },
            {
                "path": "outputs/mine_VID20260903151208_loop/reconstruction.ply",
                "name": "📹 自定义视频 1 - 回环优化 (VID151208, 196万点)",
                "category": "My Videos",
                "description": "data/mine 第1段视频 (223 帧)，196万点三维致密点云"
            },
            {
                "path": "outputs/mine_VID20260903151208_noloop/reconstruction.ply",
                "name": "📹 自定义视频 1 - 原始流式 (VID151208, 196万点)",
                "category": "My Videos",
                "description": "data/mine 第1段视频，纯因果流式预测"
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
                "id": "data/mine/VID20260903153130",
                "name": "📹 用户实拍视频 3 (VID153130)",
                "path": "data/mine/VID20260903153130",
                "description": "data/mine 实拍视频 (352 帧)"
            },
            {
                "id": "data/mine/VID20260903153215",
                "name": "📹 用户实拍视频 4 (VID153215)",
                "path": "data/mine/VID20260903153215",
                "description": "data/mine 实拍视频 (275 帧)"
            },
            {
                "id": "data/mine/VID20260903151208",
                "name": "📹 用户实拍视频 1 (VID151208)",
                "path": "data/mine/VID20260903151208",
                "description": "data/mine 实拍视频 (223 帧)"
            },
            {
                "id": "data/mine/VID20260903151228",
                "name": "📹 用户实拍视频 2 (VID151228)",
                "path": "data/mine/VID20260903151228",
                "description": "data/mine 实拍视频 (298 帧)"
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

    def handle_sse_stream(self, query_string: str) -> None:
        """Stream frame-by-frame 3D reconstruction using Server-Sent Events (SSE)."""
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

        print(f"[Streaming Server] Starting SSE stream: {image_dir_name} ({len(image_paths)} frames, stride={frame_stride})...", flush=True)
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
            except (BrokenPipeError, ConnectionResetError):
                return False

        # Send start event
        if not send_sse("start", {
            "total_frames": len(image_paths),
            "point_stride": point_stride,
            "confidence_threshold": conf_thresh,
        }):
            return

        engine = get_engine()
        with _ENGINE_LOCK:
            engine.reset()
            total_points = 0
            for i, img_path in enumerate(image_paths):
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
                    print(f"[Streaming Server] Client disconnected at frame {i}", flush=True)
                    break

                time.sleep(0.01)

            # Send complete event
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
