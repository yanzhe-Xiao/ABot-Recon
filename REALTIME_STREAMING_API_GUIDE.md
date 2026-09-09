# ABot-Recon 实时视频流点云重建并发服务接口指南

本文档介绍针对实时录制/多摄像机视频流的三维点云因果流式重建接口服务：
- **服务程序**：`viewer/streaming_api_server.py`
- **并发客户端测试工具**：`scripts/test_stream_client.py`
- **默认服务协议与端口**：`ws://<IP>:8090/ws/stream` 与 `http://<IP>:8090/`

---

## 一、 系统架构与并发机制

针对实际应用中**多路视频同时推流、并行实时建图**的业务需求，接口基于 FastAPI + WebSockets 设计了显存零冗余的高并发会话隔离架构：

```
摄像机 A (视频流 1) ──[持久化 WebSocket 连接]──► ┌──────────────────────────────────────────────┐
                                               │      MultiSessionReconstructionManager       │
摄像机 B (视频流 2) ──[持久化 WebSocket 连接]──► │                                              │
                                               │  ┌─────────────────┐    ┌─────────────────┐  │
摄像机 N (视频流 N) ──[持久化 WebSocket 连接]──► │  │ Session A 状态  │    │ Session B 状态  │  │
                                               │  │  - paged_kv (A) │    │  - paged_kv (B) │  │
                                               │  │  - camera_state │    │  - camera_state │  │
                                               │  │  - 独立点云缓存 │    │  - 独立点云缓存 │  │
                                               │  └────────┬────────┘    └────────┬────────┘  │
                                               │           └───────┬──────────────┘           │
                                               │                   ▼                          │
                                               │   [共享 GPU 模型权重 (只读常量，显存仅占 ~2GB)]  │
                                               │                   │                          │
                                               │                   ▼                          │
                                               │       逐帧流式回传 (4x4 位姿 + 增量三维点云)     │
                                               └──────────────────────────────────────────────┘
```

### 1. 显存零冗余（权重共享）
ABot-Recon 神经流式建图模型（约 1.9GB 显存，bf16）作为只读共享常量常驻 GPU。无论并发连接多少个客户端，模型权重均不需要重复复制。

### 2. 状态完全隔离（Session Isolation）
每个客户端连接时携带独立的 `session_id`。系统为每个会话分配独立的：
- **`PagedKVCacheManager`**：独立的注意力 KV-Cache 页面管理器；
- **`ref_hidden`**：专属的第一帧空间隐层表征；
- **`camera_state`**：专属性的自回归位姿追踪状态；
- **增量点云缓冲区**：独立的位姿列表与三维坐标集。
多路视频流在 GPU 上交叉调度前向计算，轨迹与空间点云互不干扰。

---

## 二、 动态可调参数清单

建立连接或发送请求时，客户端可针对每台摄像机的性能和带宽自由调节参数：

| 参数字段 | 类型 | 默认值 | 说明与调节建议 |
| :--- | :--- | :--- | :--- |
| **`session_id`** | `str` | *必填* | 客户端会话唯一 ID（如 `cam_01`, `robot_front`），用于区分不同视频流 |
| **`point_stride`** | `int` | `4` | **点采样步长**：`4`（每帧 ~8.8k 点，推荐，延迟低流畅）；`2`（每帧 ~3.5万点）；`1`（每帧 14.1万全分辨率高精度） |
| **`frame_stride`** | `int` | `1` | **时间抽帧步长**：`1` 逐帧处理；`2` 每隔 1 帧处理一次（处理 30 FPS 高帧率视频时可降低服务器负载） |
| **`voxel_size`** | `float` | `0.015` | **最终导出的体素网格大小**（单位米，如 `0.015` 表示 1.5cm 体素滤波，消除多帧重影；传 `0` 表示不去重） |
| **`confidence_threshold`** | `float` | `0.1` | **置信度阈值**（`0.0 ~ 1.0`）：滤除低置信度噪点与飞点 |
| **`include_points`** | `bool` | `true` | 是否在每帧的实时响应中返回三维坐标和 RGB 数组（若为 `false` 则只回传相机位姿与统计，节省下行带宽） |

---

## 三、 视频流结束（End-of-Stream, EOS）协议定义

为满足不同客户端平台的实现习惯，服务端提供了**四种标准化的流结束标记方式**（客户端任意使用一种均可触发服务端的收尾、体素去重与资产存盘）：

### 1. 方式一：发送文本 JSON 结束指令（最通用，推荐）
客户端发送完最后一帧图像后，直接在当前的 WebSocket 连接中发送一条 JSON 文本消息：
```json
{"type": "EOS"}
```
*(同时支持 `{"action": "end"}`、`{"action": "finish"}`、`{"command": "stop"}`)*

### 2. 方式二：发送二进制 EOS 标记（适用于纯二进制模式）
如果客户端采用全二进制帧传输，无需在文本和二进制模式之间切换，可直接发送 5 字节的魔数标记：
```python
await websocket.send(b"EOS\x00\x00")
```

### 3. 方式三：优雅断开 WebSocket 连接（网络掉线保护）
客户端调用 `websocket.close(code=1000)` 正常关闭连接时，服务端内部的 `finally` 异常守卫会自动捕获并触发点云固化，防止因客户端断线造成数据丢失。

### 4. 方式四：调用 REST 显式结束接口
```http
POST /api/stream/end?session_id=cam_01
```

---

## 四、 流结束后的服务端行为与资产导出

收到流结束标记后，服务端会自动执行以下流水线：
1. **点云空间体素去重**：聚合并使用设定的 `voxel_size`（如 1.5cm）做体素滤波，自动剔除摄像机停留/慢速移动时的双层重影；
2. **导出持久化资产**：
   - 3D 点云文件：`outputs/streams/<session_id>/reconstruction.ply`
   - 相机位姿轨迹：`outputs/streams/<session_id>/camera_poses.npy`
   - 会话统计摘要：`outputs/streams/<session_id>/session_summary.json`
3. **返回最终完成报表**：
```json
{
  "type": "session_completed",
  "session_id": "cam_alpha",
  "created_at": 1725877140.12,
  "duration_seconds": 5.71,
  "total_frames_received": 20,
  "total_frames_processed": 20,
  "raw_point_count": 118047,
  "dedup_point_count": 24620,
  "voxel_size_m": 0.015,
  "point_stride": 4,
  "frame_stride": 1,
  "avg_fps": 8.7,
  "deliverables": {
    "ply_path": "/home/data/xyz/ABot-Recon/outputs/streams/cam_alpha/reconstruction.ply",
    "ply_url": "/api/streams/cam_alpha/reconstruction.ply",
    "trajectory_path": "/home/data/xyz/ABot-Recon/outputs/streams/cam_alpha/camera_poses.npy"
  }
}
```
4. **释放显存**：自动回收并注销该会话的 KV 缓存页面。

---

## 五、 客户端调用示例代码 (Python)

### 1. WebSocket 双向流式传输客户端示例

```python
import asyncio
import json
import websockets

async def run_camera_stream():
    url = (
        "ws://127.0.0.1:8090/ws/stream?"
        "session_id=front_camera&"
        "point_stride=4&"
        "voxel_size=0.015&"
        "confidence_threshold=0.1"
    )

    async with websockets.connect(url) as ws:
        # 1. 接收服务端握手确认
        handshake = json.loads(await ws.recv())
        print("连接成功，服务端状态:", handshake["status"])

        # 2. 循环录制/读取视频帧并逐帧发送
        for frame_idx, frame_bytes in enumerate(read_camera_frames()):
            # 直接发送原始二进制 JPEG 字节（最快）
            await ws.send(frame_bytes)

            # 实时获取服务端回传的当前帧位姿与点云信息
            resp = json.loads(await ws.recv())
            pose = resp["camera_pose"]  # 4x4 旋转平移矩阵
            print(f"帧 {resp['frame_idx']:03d}: 平移 t=[{pose[0][3]:.2f}, {pose[1][3]:.2f}, {pose[2][3]:.2f}], 增量点数={resp['incremental_point_count']}")

        # 3. 视频录制结束：发送显式 EOS 标记
        print("视频结束，发送 EOS 标记...")
        await ws.send(json.dumps({"type": "EOS"}))

        # 4. 接收最终建图模型交付路径
        summary = json.loads(await ws.recv())
        print(f"建图完成！最终点云路径: {summary['deliverables']['ply_path']}")
        print(f"体素去重后点数: {summary['dedup_point_count']:,} 点")

asyncio.run(run_camera_stream())
```

---

## 六、 REST HTTP 接口概览（非 WebSocket 备用通道）

对于不支持 WebSocket 的 HTTP 客户端，服务同样提供了标准 REST 接口：

| HTTP 方法 | 接口路径 | 功能说明 |
| :--- | :--- | :--- |
| `POST` | `/api/stream/start?session_id=<id>&point_stride=4&voxel_size=0.015` | 初始化并创建会话 |
| `POST` | `/api/stream/frame?session_id=<id>` | 上传单帧原始图片二进制数据，返回当前帧位姿与点云 |
| `POST` | `/api/stream/end?session_id=<id>` | 显式通知该视频流结束，触发点云去重与 PLY 导出 |
| `GET` | `/api/sessions` | 查询当前正在推流的活跃会话以及已完成的历史会话 |
| `GET` | `/api/streams/{session_id}/reconstruction.ply` | 直接下载已完成会话导出的二进制 PLY 点云 |

---

## 七、 多客户端并发实测验证

通过 `scripts/test_stream_client.py` 模拟 **两台摄像机（`cam_alpha` 与 `cam_beta`）以不同采样步长和体素大小并发推流**：
- **`cam_alpha`**：输入视频 06，参数 `point_stride=4, voxel_size=0.015m`，结束时发送 JSON `{"type": "EOS"}`；
- **`cam_beta`**：输入视频 08，参数 `point_stride=2, voxel_size=0.020m`，结束时发送二进制 `b"EOS\x00\x00"`；

两路视频在同一张 GPU 上交替推断，帧率维持在 ~8.5 FPS，各自轨迹追踪独立闭环，视频流结束后分别导出了高质量无冲突的三维点云模型。
