<div align="center">

# ABot-Recon

## Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction

[English](README.md) | [中文](README_ZH.md)

[![Arxiv](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=5B6F9A&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.27529)
[![PDF](https://img.shields.io/static/v1?label=Paper&message=PDF&color=6A83A8&logo=adobeacrobatreader&logoColor=white)](https://github.com/amap-cvlab/ABot-Recon/blob/main/ABot-Recon-Tech-Report.pdf)
[![Project](https://img.shields.io/static/v1?label=Project&message=Website&color=2F7F83&logo=googlechrome&logoColor=white)](https://amap-cvlab.github.io/ABot-Recon-html)
[![Code](https://img.shields.io/static/v1?label=Code&message=GitHub&color=333333&logo=github&logoColor=white)](https://github.com/amap-cvlab/ABot-Recon)
[![Hugging Face](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=Hugging%20Face&color=7867A8)](https://huggingface.co/acvlab/ABot-Recon)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=5578B8)](https://modelscope.cn/models/amap_cvlab/ABot-Recon)
[![Online Demo](https://img.shields.io/static/v1?label=%F0%9F%8C%90%20Online%20Demo&message=ModelScope&color=328C8C)](https://modelscope.cn/studios/amap_cvlab/ABot-Recon)
[![Online Demo](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Online%20Demo&message=Hugging%20Face&color=7867A8)](https://huggingface.co/spaces/acvlab/abot-recon-streaming-3d)
[![License](https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=438A68)](LICENSE)

</div>

<p align="center">
  <img src="assets/teaser.png" width="85%" alt="ABot-Recon 长序列重建展示">
</p>

> **一句话介绍：** ABot-Recon 仅使用固定的 12 帧局部上下文处理超长视频流，将当前帧几何与相邻帧相对位姿逐步组合为全局重建，无需持久化的学习式长程记忆。

---

## 📣 最新功能与系统升级亮点

- **原生 WebGL2 3DGS 渲染器：** 在 8088 网页可视化端内置基于 WebGL2 的 3D Gaussian Splatting 光栅化着色器，无需第三方复杂依赖即可在浏览器中流畅交互渲染百万级高斯椭球，支持点云与 3DGS 视角平滑切换。
- **点云一键闭式转 3DGS 模型 (`scripts/pcd_to_3dgs.py`)：** 无需耗时数小时的神经网络反向传播训练，基于 KNN 局部协方差与球谐函数瞬时将重建点云闭式转换为标准 3DGS PLY 格式。
- **8090 实时视频流建图引擎与 8088 观察者协同桥梁：** 支持 WebSocket (`/ws/viewer`) 与 SSE (`/api/stream/live`) 观察者广播通道，外部推流时网页端零延迟实时渲染相机视锥运动轨迹与增量生长点云。
- **动态物体实时语义过滤 (YOLO-seg)：** 集成 YOLO 语义分割网络，推流过程中动态剔除行人、车辆等运动物体，保证背景地图纯净无拖影。
- **Method 2 多流增量场景融合与点云优化：** 支持多台机器人/多段视频协同建图，采用 Sim(3) 位姿图优化 (PGO)、多尺度 VGICP 精细配准与统计滤波 (SOR) 消除重叠伪影。

---

## 🚀 迁移部署全流程指南 (Migration & Deployment Guide)

本指南面向全新服务器环境的从零部署与现有服务的无缝迁移。

### 1. 硬件与系统环境要求

| 维度 | 最低配置要求 | 生产推荐配置 |
|---|---|---|
| **操作系统** | Linux (Ubuntu 20.04 / 22.04 LTS x86_64) | Ubuntu 22.04 LTS |
| **GPU 算力** | NVIDIA GPU (Compute Capability $\ge 8.0$：RTX 3090 / 4090 / A10 / A100 / H100) | NVIDIA RTX 4090 (24GB) 或 A100 (40GB/80GB) |
| **显存 (VRAM)** | 8 GB (仅运行离线视频推理) | 16 GB ~ 24 GB (同时运行 8090 实时推流 + YOLO 动态分割 + 8088 WebGL2 3DGS 可视化) |
| **CPU / 内存** | 8 核心 CPU，16 GB 内存 | 16+ 核心 CPU，32 GB ~ 64 GB 内存 |
| **磁盘存储** | $\ge 25\text{ GB}$ 可用 SSD 空间 (模型权重 + 缓存 + 重建产物) | NVMe 高速 SSD $\ge 100\text{ GB}$ |
| **软件环境** | Python 3.10 ~ 3.11, CUDA 12.1 | Python 3.11, CUDA 12.1, PyTorch 2.5.1 |

---

### 2. 环境搭建与依赖安装

#### 第一步：创建 Conda 独立虚拟环境
```bash
conda create -n abot-recon python=3.11 -y
conda activate abot-recon
```

#### 第二步：安装 PyTorch 与 CUDA 12.1 运行时
```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

#### 第三步：安装项目全量依赖
通过项目根目录下的 [requirements.txt](requirements.txt) 或 editable package 一键安装：

```bash
# 方式 A：通过 requirements.txt 安装 (服务器部署推荐)
pip install -r requirements.txt

# 方式 B：通过包管理安装全量扩展
pip install -e ".[all]"
```

#### 第四步：编译底层硬件加速扩展 (FlashInfer & cuRoPE)
ABot-Recon 深度利用 FlashInfer 的分页 KV-cache 显存加速机制与自定义 CUDA 旋转位置编码 (cuRoPE)：

```bash
# 1. 安装 FlashInfer 分页 KV-cache 库
pip install flashinfer-python
flashinfer show-config

# 2. 编译 cuRoPE 硬件加速算子
cd abot_recon/modeling/pi3/models/curope
pip install ninja
python setup.py build_ext --inplace
cd -
```

---

### 3. 模型权重与资产部署

系统依赖的主干权重与可选辅助模型分布如下：

```text
ABot-Recon/
├── checkpoints/
│   ├── abot_recon.safetensors         # [核心必需] ABot-Recon 重建大模型 (~4.0 GB)
│   └── loop/                          # [可选] 回环检测与特征检索权重
│       ├── dino_salad.ckpt            # (~352 MB)
│       └── dinov2_vitb14_pretrain.pth # (~346 MB)
├── yolo11m-seg.pt                     # [可选] 动态物体语义分割模型 (~45 MB)
└── yolo11n-seg.pt                     # [可选] 轻量级动态分割模型 (~6 MB)
```

#### 下载模型权重

* **ABot-Recon 核心权重 (~4.0 GB)**：
  - **Hugging Face**：[acvlab/ABot-Recon](https://huggingface.co/acvlab/ABot-Recon)
  - **ModelScope 魔搭社区**：[amap_cvlab/ABot-Recon](https://modelscope.cn/models/amap_cvlab/ABot-Recon)

  ```bash
  mkdir -p checkpoints

  # 使用 huggingface-cli 下载
  huggingface-cli download acvlab/ABot-Recon abot_recon.safetensors --local-dir checkpoints --local-dir-use-symlinks False

  # 或使用 ModelScope 下载 (国内网络极速推荐)
  python -c "from modelscope import snapshot_download; snapshot_download('amap_cvlab/ABot-Recon', local_dir='checkpoints')"
  ```

* **可选回环检索权重 (~700 MB)**：
  ```bash
  python scripts/download_loop_assets.py --output-dir checkpoints/loop
  ```

* **可选 YOLO-seg 动态过滤模型 (~45 MB)**：
  ```bash
  # 首次使用时自动下载，亦可手动预热下载：
  python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
  ```

> [!TIP]
> **关于网络代理环境配置**：如果目标服务器下载 Hugging Face 权重时遇到网络受限，可配置网络代理环境（如执行 `proxy_download` 或配置 `export HTTP_PROXY=... HTTPS_PROXY=...`），或直接通过 ModelScope 国内镜像源下载。

---

### 4. 双服务架构与一键运维管理

ABot-Recon 采用前后端分离的双服务协同架构：

```
┌────────────────────────────────────────────────────────┐
│               ABot-Recon 双服务协同架构                │
├─────────────────────────┬──────────────────────────────┤
│  8088 端口：Web 可视化服务端 │  8090 端口：在线推流建图服务端 │
│  - 原生 WebGL2 3DGS 渲染器 │  - 多会话在线流式 3D 重建    │
│  - 点云/轨迹三维交互视窗   │  - WebSocket / SSE 广播通道  │
│  - 跨端口代理 8090 数据    │  - YOLO 动态物体过滤         │
│  - 场景模型统一管理与切换  │  - 多流增量配准与融合        │
└─────────────────────────┴──────────────────────────────┘
```

通过内置的一键服务管理脚本 [`manage_services.sh`](manage_services.sh) 统一调度后台进程，已自动集成 `setsid` 与进程守护，防止 SSH 断开导致服务退出。

#### 服务管理常用命令

```bash
# 1. 一键启动所有服务 (同时启动 8088 与 8090)
./manage_services.sh start all

# 2. 查看各服务运行状态、监听端口与 PID
./manage_services.sh status

# 3. 实时滚动查看服务日志
./manage_services.sh log 8088    # 查看 8088 Web 可视化日志
./manage_services.sh log 8090    # 查看 8090 推流计算引擎日志

# 4. 重启或停止服务
./manage_services.sh restart all
./manage_services.sh stop all
```

#### 防火墙与 Nginx 反向代理配置

若服务器开启了防火墙，请放行 **8088** 与 **8090** 端口：
```bash
sudo ufw allow 8088/tcp
sudo ufw allow 8090/tcp
```

若配置 **Nginx** 反向代理并启用 HTTPS，务必开启 WebSocket 握手穿透配置：
```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:8090;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400s;
}
```

---

### 5. 实时摄像头与视频推流在线建图

使用推流工具脚本 [`scripts/stream_camera_to_8090.py`](scripts/stream_camera_to_8090.py)，可将本地 USB 摄像头、RTSP 网络视频流或离线视频实时推送到 8090 服务端：

```bash
# 1. 采集本机 USB 摄像头 (设备编号 0) 边拍边建
python scripts/stream_camera_to_8090.py --source 0 --session-id office_cam --fps 12

# 2. 接收网络 RTSP / IP 摄像头视频流
python scripts/stream_camera_to_8090.py --source "rtsp://admin:password@192.168.1.100:554/stream" --session-id robot_camera

# 3. 模拟真实拍摄帧率推送本地已有视频
python scripts/stream_camera_to_8090.py --source data/mine/video1.mp4 --session-id video1_stream --fps 15
```

**实时在线观察**：打开浏览器访问 `http://<服务器IP>:8088`，点击【8090 连接的 Stream 视频流】选项卡，在下拉列表中选择对应的会话 ID，即可在浏览器中实时看到相机视锥运动轨迹与增量生长的三维点云。

---

### 6. 点云转 3DGS 高斯模型与 WebGL2 原生渲染

使用闭式解算工具 [`scripts/pcd_to_3dgs.py`](scripts/pcd_to_3dgs.py)，可将任何点云重建结果转换为标准 3D Gaussian Splatting PLY 模型：

```bash
python scripts/pcd_to_3dgs.py \
  --input outputs/demo/reconstruction.ply \
  --output outputs/demo/reconstruction_3dgs.ply \
  --knn 16 \
  --opacity 0.85
```

- **WebGL2 网页交互渲染**：在 `http://<服务器IP>:8088` 界面中选择加载生成的 3DGS PLY 文件，点击顶部的 **3DGS (Splats)** 开关，即可享受基于椭球光栅化的高保真连续三维高斯渲染，支持实时调节椭球缩放与透明度。

---

### 7. 多视角与多流场景融合管线 (Method 2)

针对多机协同或多次扫图场景，利用 Method 2 融合工具消除各会话间的尺度漂移与局部变形：

```bash
python scripts/match_and_fuse_method2.py \
  --models outputs/stream1/reconstruction.ply outputs/stream2/reconstruction.ply \
  --output-dir outputs/fused_scene \
  --pgo \
  --sor
```

---

## 💻 离线推理与 Python API

```python
from pathlib import Path
from abot_recon import ABotRecon

images = sorted(Path("examples/images").glob("*.jpg"))

model = ABotRecon.from_pretrained(
    "acvlab/ABot-Recon",
    device="cuda",
    attention_backend="auto",
    loop_closure=False,
)

result = model.infer(images)

trajectory = result.camera_poses      # 相机轨迹位姿
relative_poses = result.relative_poses # 相邻帧相对位姿
local_points = result.local_points     # 逐帧局部点图
confidence = result.confidence         # 深度置信度
```

导出带颜色的稠密点云 PLY 与俯视轨迹图：
```bash
python scripts/export_reconstruction_ply.py \
  --poses outputs/demo/camera_poses.npy \
  --points outputs/demo/local_points.pt \
  --colors outputs/demo/colors.pt \
  --output outputs/demo/reconstruction.ply \
  --bev-output outputs/demo/trajectory_bev.png
```

---

## 🛠️ 迁移与部署排错排查清单 (Troubleshooting)

| 异常现象 | 可能原因 | 解决办法 |
|---|---|---|
| **CUDA Out of Memory (显存溢出)** | 序列过长或显存不足以同时支撑推理与分割 | 1. 启动时增加 `--dense-stride 2` 或推流时增加 `point_stride=4`<br>2. 临时关闭动态过滤：不传 `--dynamic-filter`<br>3. 降低推流分辨率（如 504×280） |
| **FlashInfer ImportError 或 ABI 冲突** | FlashInfer 编译环境与当前 PyTorch/CUDA 不兼容 | 启动时添加 `--attention-backend sdpa`，系统将自动回退到 PyTorch 原生高效率注意力机制 |
| **cuRoPE 编译失败** | 缺少 `ninja` 或 GCC 编译器版本过低 | 执行 `pip install ninja` 并检查 `gcc --version` ($\ge 9.0$)。cuRoPE 为性能加速项，编译失败会自动安全回退 |
| **8088 / 8090 端口已被占用** | 残留的旧后台进程未彻底退出 | 执行 `./manage_services.sh stop all` 或通过 `lsof -i :8088` 查询 PID 后执行 `kill -9 <PID>` 释放端口 |
| **8090 推流 WebSocket 无法连接或立即断开** | 反向代理未正确升级 WebSocket 握手协议 | 在 Nginx 配置文件中补齐 `proxy_set_header Upgrade $http_upgrade;` 与 `proxy_set_header Connection "upgrade";` |
| **Hugging Face 权重下载缓慢或超时** | 跨境网络不稳定或受限 | 使用 ModelScope 镜像源下载：`python -c "from modelscope import snapshot_download; snapshot_download('amap_cvlab/ABot-Recon', local_dir='checkpoints')"` |

---

## 🧪 单元与集成测试

```bash
# 执行基础单元测试
pytest -q

# 验证 cuRoPE CUDA 算子与 PyTorch 等价性
ABOT_RECON_REQUIRE_CUROPE=1 pytest -q tests/test_curope_parity.py

# 验证真实模型权重推理
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_IMAGE_DIR=examples/images \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_checkpoint.py
```

---

## 📖 引用

```bibtex
@misc{han2026revisitinglocalcontextlonghorizon,
      title={Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction}, 
      author={Jiarong Han and Jincheng Xiong and Yuzhou Liu and Linzhe Shi and Changjie Wu and Ning Guo and Mu Xu and Hang Zhang and Ming Qian},
      year={2026},
      eprint={2608.27529},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.27529}, 
}
```

---

## 📄 开源许可证与致谢

本项目源码遵循 [Apache License 2.0](LICENSE) 许可。模型权重的使用受 [MODEL_LICENSE.md](MODEL_LICENSE.md) 约束，第三方依赖说明参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

使用模型前请仔细阅读 [模型使用规范 (Model Usage Guidelines)](MODEL_USAGE_GUIDELINES.md)。

ABot-Recon 基于 Pi3 构建，并从 CroCo、DUSt3R、DINOv2、SALAD、FlashInfer、LingBot-Map、HorizonStream 和 LongStream 等优秀开源工作中获得启发，特此向相关作者与社区贡献者致谢。

同时衷心感谢葛增叶、潘宏宇、孙忠旭、汪奔涛、徐玉婷、欧阳天健、余浩铭、陈楚子与张智阳在本项目推进过程中给予的宝贵支持与贡献。

---

## 🌐 团队其他工作

- [ABot-Earth](https://abot-earth.amap.com/)
