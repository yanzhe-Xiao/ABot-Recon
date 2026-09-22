#!/usr/bin/env python3
"""
stream_camera_to_8090.py
实时推流拍摄工具：将摄像头（USB/手机）或视频文件实时推送到 8090 端口，
同时在 8088 端口的 Web 界面上可以看到实时 3D 点云建图与相机位姿运动轨迹。

用法示例：
1. 捕获默认本机摄像头边拍边建：
   python scripts/stream_camera_to_8090.py --source 0 --session-id my_camera

2. 模拟真实拍摄速度推送本地视频文件：
   python scripts/stream_camera_to_8090.py --source /path/to/video.mp4 --fps 12

3. 推送网络 RTSP / IP 摄像头视频流：
   python scripts/stream_camera_to_8090.py --source rtsp://admin:12345@192.168.1.100:554/stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_source(source_str: str):
    """Parse source string to int (camera index) or str (video file/stream)."""
    if source_str.isdigit():
        return int(source_str)
    return source_str


async def stream_worker(
    source,
    server_ws: str,
    session_id: str,
    fps: float = 12.0,
    width: int = 504,
    height: int = 280,
    max_frames: int = 0,
    loop: bool = False,
    show_preview: bool = False,
    max_in_flight: int = 2,
):
    # Open VideoCapture
    print(f"[*] 正在打开视频源: {source} ...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频源: {source}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"[+] 视频源打开成功! 原始尺寸: {orig_w}x{orig_h}, FPS: {src_fps:.1f}")
    print(f"[+] 推流目标分辨率: {width}x{height}, 目标推流帧率: {fps:.1f} FPS, 流水线深度: {max_in_flight}")

    # Connect to 8090 WebSocket stream endpoint
    url = (
        f"{server_ws}/ws/stream?"
        f"session_id={session_id}&"
        f"point_stride=4&"
        f"frame_stride=1&"
        f"voxel_size=0.015&"
        f"confidence_threshold=0.1&"
        f"include_points=false"
    )

    print(f"\n{'='*65}")
    print(f"📡 正在连接 8090 推流引擎: {url}")
    print(f"💡 提示：打开浏览器访问 http://<你的IP>:8088")
    print(f"   在【8090 连接的 Stream 视频流】下拉框选择 [{session_id}] 即可实时查看 3D 建图！")
    print(f"{'='*65}\n")

    delay = 1.0 / max(1.0, fps)
    frames_sent = 0
    frames_received = 0
    t_start = time.time()

    try:
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            # Receive initial handshake ack
            handshake_raw = await ws.recv()
            handshake = json.loads(handshake_raw)
            print(f"[+] 8090 服务端握手成功: {handshake}")

            sem = asyncio.Semaphore(max_in_flight)
            summary_container: dict = {}
            eos_received = asyncio.Event()

            async def receiver():
                nonlocal frames_received, summary_container
                while not eos_received.is_set():
                    try:
                        resp_raw = await ws.recv()
                    except Exception:
                        break
                    try:
                        resp = json.loads(resp_raw)
                    except Exception:
                        continue

                    msg_type = resp.get("type")
                    if msg_type == "frame_result":
                        sem.release()
                        frames_received += 1
                        pose = resp.get("camera_pose", [])
                        t_xyz = [round(pose[i][3], 2) for i in range(3)] if len(pose) >= 3 else [0, 0, 0]
                        pts = resp.get("incremental_point_count", 0)
                        inf_ms = resp.get("inference_time_ms", 0)
                        r_fps = resp.get("fps", 0)
                        f_idx = resp.get("frame_idx", frames_received - 1)
                        dev = resp.get("worker_device", "")
                        dev_str = f" [{dev}]" if dev else ""
                        print(
                            f"\r[Session {session_id}] 帧 #{f_idx+1:04d} | 相机位置: {t_xyz} | "
                            f"新增点数: +{pts:4d} | 耗时: {inf_ms:4.1f}ms ({r_fps:.1f} FPS){dev_str} | 8088 实时建图中...",
                            end="",
                            flush=True,
                        )
                    elif msg_type == "session_completed":
                        summary_container.update(resp)
                        eos_received.set()
                        break

            recv_task = asyncio.create_task(receiver())

            while True:
                ret, frame = cap.read()
                if not ret:
                    if loop and not isinstance(source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        print("\n[*] 视频源播放完毕 (EOF).")
                        break

                t_frame0 = time.time()

                # Resize to target resolution
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame

                # Encode to JPEG
                _, encoded_buf = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = encoded_buf.tobytes()

                # Acquire pipeline permit to avoid unbounded frame queuing
                await sem.acquire()
                # Send binary JPEG frame over WebSocket asynchronously
                await ws.send(jpeg_bytes)
                frames_sent += 1

                if show_preview:
                    cv2.putText(
                        frame_resized,
                        f"Live 8090 Stream: #{frames_sent} | Session: {session_id}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.imshow("Camera Stream to 8090 (Press Q to quit)", frame_resized)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[!] 用户中断视频捕获.")
                        break

                if max_frames > 0 and frames_sent >= max_frames:
                    print(f"\n[*] 已达到最大帧数限制: {max_frames}")
                    break

                # Control frame rate smoothly
                elapsed_frame = time.time() - t_frame0
                sleep_sec = delay - elapsed_frame
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

            # Send EOS
            print(f"\n[*] 正在发送 End-of-Stream (EOS) 结束信号...")
            await ws.send(json.dumps({"type": "EOS"}))

            # Wait for receiver to finish receiving completion summary
            try:
                await asyncio.wait_for(recv_task, timeout=30.0)
            except asyncio.TimeoutError:
                print("\n[!] 等待 EOS 汇总响应超时")

            summary = summary_container
            total_duration = time.time() - t_start
            print("\n" + "=" * 65)
            print("🎉 3D 建图会话结束！交付成果清单:")
            print(f" - 总推流帧数: {frames_sent} 帧 (已确认: {frames_received} 帧)")
            print(f" - 总建图点数: {summary.get('total_points', 0):,} 点")
            print(f" - 耗时: {total_duration:.1f} 秒 (实际推流 FPS: {frames_sent / max(0.1, total_duration):.1f})")
            print(f" - PLY 点云保存路径: {summary.get('deliverables', {}).get('ply_path', summary.get('ply_path', 'N/A'))}")
            print(f" - 8088 场景查看地址: http://127.0.0.1:8088/outputs/streams/{session_id}/stream_reconstruction.ply")
            print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[Error] 推流过程出错: {e}")
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="实时推流拍摄工具：将摄像头或视频推送到 8090，并在 8088 查看实时 3D 建图",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="视频源：摄像头索引 (如 0, 1) 或 视频文件路径 (如 /path/to/video.mp4) 或 RTSP URL",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="ws://127.0.0.1:8090",
        help="8090 WebSocket 基础地址 (默认: ws://127.0.0.1:8090)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=f"stream_{int(time.time())}",
        help="推流会话唯一标识符 (默认自动生成带时间戳的 ID)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="模拟拍摄/推流帧率 (默认: 12.0 FPS)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=504,
        help="输入图像缩放宽度 (默认: 504)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=280,
        help="输入图像缩放高度 (默认: 280)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最大推流帧数 (0 表示不限制)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="若输入为视频文件，播放到结尾时是否自动循环",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="是否开启本地 OpenCV 视频预览窗口",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=2,
        help="异步流水线最大在途帧数 (默认: 2，有效解除停等延迟瓶颈)",
    )

    args = parser.parse_args()
    source = parse_source(args.source)

    asyncio.run(
        stream_worker(
            source=source,
            server_ws=args.server,
            session_id=args.session_id,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            loop=args.loop,
            show_preview=args.preview,
            max_in_flight=args.max_in_flight,
        )
    )


if __name__ == "__main__":
    main()
