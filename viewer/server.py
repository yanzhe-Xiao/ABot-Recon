#!/usr/bin/env python3
"""HTTP & Server-Sent Events (SSE) Streaming Server for ABot-Recon 3D Visualizer."""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from viewer.stream_backend import OnlineReconstructionEngine

ROOT_DIR = Path(__file__).resolve().parent.parent

# Global singleton engine
_ENGINE: Optional[OnlineReconstructionEngine] = None


def get_engine() -> OnlineReconstructionEngine:
    global _ENGINE
    if _ENGINE is None:
        print("[Streaming Server] Initializing ABot-Recon Streaming Engine...")
        _ENGINE = OnlineReconstructionEngine(
            checkpoint=ROOT_DIR / "checkpoints/abot_recon.safetensors",
            device="cuda",
            confidence_threshold=0.1,
            point_stride=4,
        )
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

        if path == "/api/status":
            return self.handle_status()

        return super().do_GET()

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
        image_dir_name = params.get("image_dir", ["examples/images"])[0]

        image_dir = ROOT_DIR / image_dir_name
        if not image_dir.is_dir():
            self.send_error(404, f"Image directory {image_dir_name} not found")
            return

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )[::frame_stride]

        if not image_paths:
            self.send_error(404, "No image frames found in directory")
            return

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
        engine.reset()

        total_points = 0
        for i, img_path in enumerate(image_paths):
            res = engine.process_frame(
                img_path,
                confidence_threshold=conf_thresh,
                point_stride=point_stride,
            )
            total_points += res["point_count"]

            frame_data = {
                "frame_index": res["frame_index"],
                "total_frames": len(image_paths),
                "image_url": f"/{img_path.relative_to(ROOT_DIR)}",
                "camera_pose": res["camera_pose"],
                "point_count": res["point_count"],
                "total_accumulated_points": total_points,
                "points_xyz": res["points_xyz"].tolist(),
                "points_rgb": res["points_rgb"].tolist(),
                "latency_ms": round(res["inference_time_ms"], 1),
                "fps": round(res["fps"], 1),
            }

            if not send_sse("frame", frame_data):
                print(f"[Streaming Server] Client disconnected at frame {i}")
                break

            time.sleep(0.01)

        # Send complete event
        send_sse("complete", {
            "total_frames": len(image_paths),
            "total_points": total_points,
        })

    def log_message(self, format: str, *args) -> None:
        pass


def run_server(port: int = 8088, host: str = "0.0.0.0") -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), StreamingRequestHandler) as httpd:
        print(f"[ABot-Recon 3D Streaming Server] Running at http://127.0.0.1:{port}/")
        print(f"[ABot-Recon 3D Streaming Server] Streaming API at http://127.0.0.1:{port}/api/stream")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ABot-Recon 3D streaming visualizer")
    parser.add_argument("--port", type=int, default=8088, help="Port to serve on (default: 8088)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
