#!/usr/bin/env python3
"""
Test Client: Simulates Multiple Cameras Streaming Concurrently to the ABot-Recon API.
Demonstrates persistent bidirectional WebSocket streaming, real-time pose/point feedback,
and explicit End-of-Stream (EOS) signaling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
import websockets


REPO_ROOT = Path(__file__).resolve().parent.parent


async def stream_camera(
    server_url: str,
    session_id: str,
    image_paths: list[Path],
    scene_id: Optional[str] = None,
    fps: float = 15.0,
    point_stride: int = 4,
    frame_stride: int = 1,
    voxel_size: float = 0.015,
    confidence_thresh: float = 0.1,
    auto_fuse: bool = True,
    eos_method: str = "json",  # "json" or "binary"
) -> dict:
    """Stream an image sequence as a simulated real-time camera over persistent WebSocket."""
    url = (
        f"{server_url}/ws/stream?"
        f"session_id={session_id}&"
        f"point_stride={point_stride}&"
        f"frame_stride={frame_stride}&"
        f"voxel_size={voxel_size}&"
        f"confidence_threshold={confidence_thresh}&"
        f"include_points=false"
    )
    if scene_id:
        url += f"&scene_id={scene_id}"
    url += f"&auto_fuse={'true' if auto_fuse else 'false'}"
    print(f"[{session_id}] Connecting to {url}...")
    delay = 1.0 / max(1.0, fps)

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        # 1. Receive Handshake
        handshake_raw = await ws.recv()
        handshake = json.loads(handshake_raw)
        print(f"[{session_id}] Handshake Ack: status={handshake.get('status')}")

        t_start = time.time()
        frames_sent = 0

        # 2. Continuous frame streaming
        for idx, img_path in enumerate(image_paths):
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            t_send0 = time.time()
            # Send raw binary JPEG
            await ws.send(img_bytes)
            frames_sent += 1

            # Receive per-frame real-time feedback
            resp_raw = await ws.recv()
            resp = json.loads(resp_raw)

            if resp.get("type") == "frame_result":
                pose = resp["camera_pose"]
                t_vec = [round(pose[i][3], 3) for i in range(3)]
                print(
                    f"[{session_id}] Frame {resp['frame_idx']:03d} -> Pose t={t_vec}, "
                    f"pts={resp['incremental_point_count']:,}, "
                    f"latency={resp['inference_time_ms']:.1f}ms, fps={resp['fps']:.1f}"
                )

            # Throttle to simulate real-world camera recording FPS
            sleep_time = delay - (time.time() - t_send0)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # 3. Explicit End-of-Stream (EOS) Signaling
        print(f"\n[{session_id}] Video ended! Sending End-of-Stream ({eos_method})...")
        if eos_method == "json":
            # Protocol Option 1: Send text EOS
            await ws.send(json.dumps({"type": "EOS"}))
        else:
            # Protocol Option 2: Send binary EOS token
            await ws.send(b"EOS\x00\x00")

        # 4. Receive Final Completion Summary
        summary_raw = await ws.recv()
        summary = json.loads(summary_raw)
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"[{session_id}] STREAM COMPLETED successfully!")
        print(f"  Duration:           {elapsed:.2f}s")
        print(f"  Frames processed:   {summary.get('total_frames_processed')}")
        print(f"  Raw points:         {summary.get('raw_point_count'):,}")
        print(f"  Dedup points:       {summary.get('dedup_point_count'):,}")
        print(f"  Final PLY model:    {summary.get('deliverables', {}).get('ply_path')}")
        print(f"{'='*60}\n")
        return summary


async def run_concurrent_test(server_url: str, scene_id: str = "simul_scene", num_frames: int = 20) -> None:
    """Run two camera streams concurrently in parallel, sending frames and ending simultaneously to the SAME scene."""
    paths_06 = sorted(Path("data/data/06").glob("*.jpg"))[:num_frames]
    paths_08 = sorted(Path("data/data/08").glob("*.jpg"))[:num_frames]

    print(f"Testing simultaneous multi-stream reconstruction & concurrent EOS for Scene: [{scene_id}]")
    print(f"  Camera 1 (cam_alpha): {len(paths_06)} frames from data/data/06 (point_stride=4, voxel=0.015m)")
    print(f"  Camera 2 (cam_beta):  {len(paths_08)} frames from data/data/08 (point_stride=4, voxel=0.015m)")
    print("-" * 70)

    task1 = stream_camera(
        server_url=server_url,
        session_id=f"{scene_id}_cam_alpha",
        scene_id=scene_id,
        image_paths=paths_06,
        fps=20.0,
        point_stride=4,
        voxel_size=0.015,
        auto_fuse=True,
        eos_method="json",
    )
    task2 = stream_camera(
        server_url=server_url,
        session_id=f"{scene_id}_cam_beta",
        scene_id=scene_id,
        image_paths=paths_08,
        fps=20.0,
        point_stride=4,
        voxel_size=0.015,
        auto_fuse=True,
        eos_method="binary",
    )

    results = await asyncio.gather(task1, task2)
    print("Both concurrent streams completed and ended simultaneously!")


async def run_sequential_scene_test(server_url: str, scene_id: str = "corridor_scene", num_frames: int = 25) -> None:
    """Test sequential arrival of multiple video streams contributing to the SAME scene_id."""
    paths_06 = sorted(Path("data/data/06").glob("*.jpg"))[:num_frames]
    paths_08 = sorted(Path("data/data/08").glob("*.jpg"))[:num_frames]

    print(f"\n=======================================================================")
    print(f"  Testing Sequential Stream Arrival for Scene: [{scene_id}]")
    print(f"=======================================================================")
    print(f"[Step 1] First stream (stream_01) arrives and reconstructs baseline...")
    res1 = await stream_camera(
        server_url=server_url,
        session_id=f"{scene_id}_stream_01",
        scene_id=scene_id,
        image_paths=paths_06,
        fps=20.0,
        point_stride=4,
        voxel_size=0.015,
        auto_fuse=True,
        eos_method="json",
    )
    print(f"[Step 1 Completed] Stream 01 finished: {res1.get('dedup_point_count', 0)} points.")
    print(f"  Scene [{scene_id}] current model available at: /api/scenes/{scene_id}/reconstruction.ply")

    print(f"\n[Step 2] Second stream (stream_02) arrives later into the SAME scene...")
    res2 = await stream_camera(
        server_url=server_url,
        session_id=f"{scene_id}_stream_02",
        scene_id=scene_id,
        image_paths=paths_08,
        fps=20.0,
        point_stride=4,
        voxel_size=0.015,
        auto_fuse=True,
        eos_method="binary",
    )
    print(f"[Step 2 Completed] Stream 02 finished: {res2.get('dedup_point_count', 0)} points.")
    print(f"  Scene [{scene_id}] auto-fused! Unified download: /api/scenes/{scene_id}/reconstruction.ply")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-stream and Scene API test client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--mode", choices=["concurrent", "sequential"], default="sequential", help="Test mode: sequential scene build or concurrent test")
    parser.add_argument("--scene-id", default="demo_corridor", help="Scene ID to test sequential stream building")
    args = parser.parse_args()

    server_ws = f"ws://{args.host}:{args.port}"
    if args.mode == "sequential":
        asyncio.run(run_sequential_scene_test(server_ws, scene_id=args.scene_id, num_frames=args.frames))
    else:
        asyncio.run(run_concurrent_test(server_ws, scene_id=args.scene_id, num_frames=args.frames))


if __name__ == "__main__":
    main()
