#!/usr/bin/env python3
"""Lightweight HTTP server for ABot-Recon 3D Point Cloud Web Visualizer."""

from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html", "/viewer"):
            self.path = "/viewer/index.html"
        return super().do_GET()

    def log_message(self, format: str, *args) -> None:
        # Keep logs clean
        pass


def run_server(port: int = 8088, host: str = "0.0.0.0") -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), CORSHTTPRequestHandler) as httpd:
        print(f"[ABot-Recon 3D Visualizer] Server running at http://127.0.0.1:{port}/")
        print(f"[ABot-Recon 3D Visualizer] Access from any browser: http://<your-ip>:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ABot-Recon 3D point cloud visualizer")
    parser.add_argument("--port", type=int, default=8088, help="Port to serve on (default: 8088)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
