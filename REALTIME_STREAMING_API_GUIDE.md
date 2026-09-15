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
| **`session_id`** | `str` | *必填* | 客户端单路流会话唯一 ID（如 `cam_01`, `robot_front`），用于底层推流与会话隔离 |
| **`scene_id`** | `str` | `session_id` | **所属场景唯一 ID**（如 `corridor_hall`, `building_1f`）。不同时间先后接入的视频流指定相同 `scene_id` 即可自动汇聚到同一场景 |
| **`auto_fuse`** | `bool` | `true` | 当该视频流结束时，若同一场景内已存在其他已完成视频流，是否**自动触发方案二多模态全局融合** |
| **`point_stride`** | `int` | `4` | **点采样步长**：`4`（每帧 ~8.8k 点，推荐，延迟低流畅）；`2`（每帧 ~3.5万点）；`1`（每帧 14.1万全分辨率高精度） |
| **`frame_stride`** | `int` | `1` | **时间抽帧步长**：`1` 逐帧处理；`2` 每隔 1 帧处理一次（处理 30 FPS 高帧率视频时可降低服务器负载） |
| **`voxel_size`** | `float` | `0.015` | **最终导出的体素网格大小**（单位米，如 `0.015` 表示 1.5cm 体素滤波，消除多帧重影；传 `0` 表示不去重） |
| **`confidence_threshold`** | `float` | `0.1` | **置信度阈值**（`0.0 ~ 1.0`）：滤除低置信度噪点与飞点 |
| **`dynamic_filter`** | `bool` | `false` | **动态物体实时滤除开关**：开启后实时运行 YOLO-seg 语义分割，彻底抹除画面中的行人、车辆等移动物体及三维残影 |
| **`include_points`** | `bool` | `true` | 是否在每帧的实时响应中返回三维坐标和 RGB 数组（若为 `false` 则只回传相机位姿与统计，节省下行带宽） |

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
2. **导出持久化资产与多模态匹配张量**：
   - 3D 点云文件：`outputs/streams/<session_id>/reconstruction.ply`
   - 像素 3D 反查表：`outputs/streams/<session_id>/world_points.pt`（为方案二配准必备）
   - 视频帧 RGB 张量：`outputs/streams/<session_id>/colors.pt`（为方案二特征提取必备）
   - 置信度张量：`outputs/streams/<session_id>/confidence.pt`
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

---

## 五、 多视频流先后接入与统一场景融合建图机制 (Scene-Based Fusion)

针对“**多个视频流属于同一物理场景、但在不同时刻先后接入**”的应用需求，系统现已全面升级**场景级（Scene-Level）聚合与融合机制**：

```
时刻 T1: 视频流 A (session_id=cam_A, scene_id=room_01) 接入 ──► 构建首段点云，确立基准坐标系 ──► 下载 /api/scenes/room_01/reconstruction.ply (此时含流 A)
                                                                                                          ▲
时刻 T2: 视频流 B (session_id=cam_B, scene_id=room_01) 接入 ──► 独立推断结束 ──► 触发多模态自动配准融合 ────────┤ (更新为 A+B 融合全景点云)
                                                                                                          ▲
时刻 T3: 视频流 N (session_id=cam_N, scene_id=room_01) 接入 ──► 独立推断结束 ──► 触发多模态增量配准融合 ────────┘ (更新为 A+B+...+N 超高清全景)
```

### 1. 核心特性
1. **解耦传输与流式隔离**：各视频流不论是同时传、错峰传、还是间隔数小时传输，流式服务内部计算状态完全独立，保证各流自回归跟踪位姿的纯净度；
2. **零配置增量融合**：同场景第二路及后续视频流传输完毕（EOS）时，服务端自动识别重叠纹理与空间点集，运用 `ALIKED + LightGlue + Umeyama (Sim(3)) + small_gicp` 算法推算旋转、平移与绝对尺度缩放系数，直接将多路点云无缝融合；
3. **统一场景下载端点**：上层业务系统无需记录各视频流细节，**直接使用场景 ID 即可统一获取该场景当前最新的融合点云模型**；
4. **前端无缝呈现**：已融合的场景点云会自动注册至 8088 端口的三维交互前端（`多视频流场景融合点云 (Multi-Stream Scenes)` 分组下），可随时交互巡检。

### 2. 场景相关 HTTP REST 接口

| HTTP 方法 | 接口路径 | 功能说明 |
| :--- | :--- | :--- |
| `GET` | `/api/scenes` | 列出全部已创建的场景列表、包含的视频流数量与当前融合状态 |
| `GET` | `/api/scenes/{scene_id}` | 获取特定场景的详细统计信息、包含会话列表与交付文件清单 |
| `GET` | `/api/scenes/{scene_id}/reconstruction.ply` | **直接按场景 ID 下载统一融合后的 3D 点云**（真彩色） |
| `GET` | `/api/scenes/{scene_id}/reconstruction.ply?colored=true` | **直接按场景 ID 下载多色区分点云**（各视频流赋予红/绿/蓝等高对比色） |
| `POST` | `/api/scenes/{scene_id}/fuse?voxel_size=0.015` | 显式手动触发或重新运行该场景下所有已完成视频流的融合对齐 |
| `GET` | `/api/scenes/{scene_id}/transforms.json` | 下载该场景下各视频流相对基准坐标系的 Sim(3) 尺度因子与 4x4 变换矩阵 |

---

## 六、 动态物体实时滤除与服务端启动配置 (`--dynamic-filter`)

在走廊、展厅、车间等存在行走人员或移动设备的真实场景中，实时推流建图容易产生动态人物拉丝和三维残影。服务端现已原生内置基于 YOLO-seg 语义分割的**动态物体像素级实时滤除引擎**。

### 1. 服务端启动命令行参数

启动 `viewer/streaming_api_server.py` 时，可通过以下参数控制动态滤除行为：

```bash
# 1. 默认启动（轻量模式，不启用动态滤除，低延迟）
python viewer/streaming_api_server.py --port 8090

# 2. 开启实时动态物体滤除（默认全局开启 YOLO-seg 过滤，彻底去除行人/车辆）
python viewer/streaming_api_server.py --port 8090 --dynamic-filter

# 3. 高精度模式（采用更大模型，提升遮挡人体与远距离物体的分割精度）
python viewer/streaming_api_server.py --port 8090 \
  --dynamic-filter \
  --dynamic-model yolo11m-seg.pt \
  --dynamic-conf 0.15 \
  --dynamic-dilate 11
```

| 服务端启动参数 | 类型 | 默认值 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **`--dynamic-filter`** / **`--no-dynamic-filter`** | `bool` | `False` | 是否在服务端全局默认开启动态物体实时滤除功能 |
| **`--dynamic-model`** | `str` | `yolo11n-seg.pt` | 分割模型权重，可选 `yolo11n-seg.pt`（极速 ~5ms）、`yolo11m-seg.pt`（高精度 ~18ms） |
| **`--dynamic-conf`** | `float` | `0.15` | 动态实体检测置信度阈值（建议 `0.10 ~ 0.25`） |
| **`--dynamic-dilate`** | `int` | `7` | 形态学膨胀核半径（像素），外扩掩模以彻底吸收人体衣物和边缘毛刺 |

### 2. 客户端单流动态重写（Per-Session Override）

即使服务端默认未全局开启 `--dynamic-filter`，特定客户端也可以在建立连接时通过 URL 参数主动开启：

- **WebSocket 连接参数**：
  ```text
  ws://<IP>:8090/ws/stream?session_id=cam_01&dynamic_filter=true
  ```
- **REST 启动会话参数**：
  ```http
  POST /api/stream/start?session_id=cam_01&dynamic_filter=true
  ```

### 3. 工作原理与资产交付
1. **逐帧共形对齐**：视频帧进入模型时，实时生成与 $280 \times 504$ 点图完全对齐的布尔静态掩膜；
2. **增量回传置零**：处于人体、车辆范围内的三维点被实时剔除，不参与 WebSocket 下发；
3. **存盘固化**：流结束（EOS）时，服务端不仅导出已滤除动态人物的纯净 `reconstruction.ply`，还会同步导出 `static_masks.pt`，为后续场景级多路视频融合提供纯净的多模态匹配基准。

---

## 七、 客户端调用示例代码 (Python)

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

## 八、 REST HTTP 接口概览（非 WebSocket 备用通道）

对于不支持 WebSocket 的 HTTP 客户端，服务同样提供了标准 REST 接口：

| HTTP 方法 | 接口路径 | 功能说明 |
| :--- | :--- | :--- |
| `POST` | `/api/stream/start?session_id=<id>&point_stride=4&voxel_size=0.015` | 初始化并创建会话 |
| `POST` | `/api/stream/frame?session_id=<id>` | 上传单帧原始图片二进制数据，返回当前帧位姿与点云 |
| `POST` | `/api/stream/end?session_id=<id>` | 显式通知该视频流结束，触发点云去重与 PLY 导出 |
| `POST` | `/api/fuse?session_ids=cam_1,cam_2&outputs=normal,colored` | **调用方法二对完成的流式点云执行多模态极速融合** |
| `GET` | `/api/sessions` | 查询当前正在推流的活跃会话以及已完成的历史会话 |
| `GET` | `/api/fusions` | 列出所有已生成的方案二融合模型资产清单与下载链接 |
| `GET` | `/api/streams/{session_id}/reconstruction.ply` | 直接下载已完成单流会话导出的二进制 PLY 点云 |
| `GET` | `/api/fusions/{fusion_id}/{filename}` | 直接下载方案二多流融合后的 PLY 点云或报告 |

---

## 九、 多客户端并发与时序融合实测验证

通过 `scripts/test_stream_client.py` 模拟 **两台摄像机（`cam_alpha` 与 `cam_beta`）以不同采样步长和体素大小并发推流**：
- **`cam_alpha`**：输入视频 06，参数 `point_stride=4, voxel_size=0.015m`，结束时发送 JSON `{"type": "EOS"}`；
- **`cam_beta`**：输入视频 08，参数 `point_stride=2, voxel_size=0.020m`，结束时发送二进制 `b"EOS\x00\x00"`；

两路视频在同一张 GPU 上交替推断，帧率维持在 ~8.5 FPS，各自轨迹追踪独立闭环，视频流结束后分别导出了高质量无冲突的三维点云模型。
