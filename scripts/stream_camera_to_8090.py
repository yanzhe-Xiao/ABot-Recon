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
    print(f"[+] 推流目标分辨率: {width}x{height}, 目标推流帧率: {fps:.1f} FPS")

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
    t_start = time.time()

    try:
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            # Receive initial handshake ack
            handshake_raw = await ws.recv()
            handshake = json.loads(handshake_raw)
            print(f"[+] 8090 服务端握手成功: {handshake}")

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

                # Send binary JPEG frame over WebSocket
                await ws.send(jpeg_bytes)
                frames_sent += 1

                # Wait for frame processing acknowledgment
                resp_raw = await ws.recv()
                resp = json.loads(resp_raw)

                if resp.get("type") == "frame_result":
                    pose = resp.get("camera_pose", [])
                    t_xyz = [round(pose[i][3], 2) for i in range(3)] if len(pose) >= 3 else [0, 0, 0]
                    pts = resp.get("incremental_point_count", 0)
                    inf_ms = resp.get("inference_time_ms", 0)
                    print(
                        f"\r[Session {session_id}] 帧 #{frames_sent:04d} | 相机位置: {t_xyz} | "
                        f"新增点数: +{pts:4d} | 耗时: {inf_ms:4.1f}ms | 8088 正在实时建图...",
                        end="",
                        flush=True,
                    )

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

                # Control frame rate
                elapsed_frame = time.time() - t_frame0
                sleep_sec = delay - elapsed_frame
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

            # Send EOS
            print(f"\n[*] 正在发送 End-of-Stream (EOS) 结束信号...")
            await ws.send(json.dumps({"type": "EOS"}))

            # Receive completion summary
            summary_raw = await ws.recv()
            summary = json.loads(summary_raw)
            total_duration = time.time() - t_start
            print("\n" + "=" * 65)
            print("🎉 3D 建图会话结束！交付成果清单:")
            print(f" - 总推流帧数: {frames_sent} 帧")
            print(f" - 总建图点数: {summary.get('total_points', 0):,} 点")
            print(f" - 耗时: {total_duration:.1f} 秒 (实际推流 FPS: {frames_sent / max(0.1, total_duration):.1f})")
            print(f" - PLY 点云保存路径: {summary.get('ply_path', 'N/A')}")
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
        )
    )


if __name__ == "__main__":
    main()
