<div align="center">

# ABot-Recon

## Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction

[English](README.md) | [中文](README_ZH.md)

[![Arxiv](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=5B6F9A&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.27529)
[![Tech PDF](https://img.shields.io/static/v1?label=Paper&message=PDF&color=6A83A8&logo=adobeacrobatreader&logoColor=white)](https://github.com/amap-cvlab/ABot-Recon/blob/main/ABot-Recon-Tech-Report.pdf)  
[![Project](https://img.shields.io/static/v1?label=Project&message=Website&color=2F7F83&logo=googlechrome&logoColor=white)](https://amap-cvlab.github.io/ABot-Recon-html)
[![Code](https://img.shields.io/static/v1?label=Code&message=GitHub&color=333333&logo=github&logoColor=white)](https://github.com/amap-cvlab/ABot-Recon)
[![Hugging Face](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=Hugging%20Face&color=7867A8)](https://huggingface.co/acvlab/ABot-Recon)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=5578B8)](https://modelscope.cn/models/amap_cvlab/ABot-Recon)
[![Online Demo](https://img.shields.io/static/v1?label=%F0%9F%8C%90%20Online%20Demo&message=ModelScope&color=328C8C)](https://modelscope.cn/studios/amap_cvlab/ABot-Recon)
[![Online Demo](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Online%20Demo&message=Hugging%20Face&color=7867A8)](https://huggingface.co/spaces/acvlab/abot-recon-streaming-3d)
[![License](https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=438A68)](LICENSE)

</div>

<p align="center">
  <img src="assets/teaser.png" width="85%" alt="ABot-Recon long-horizon reconstruction teaser">
</p>

> **In one sentence:** ABot-Recon reconstructs long video streams with a fixed 12-frame local context, composing current-frame geometry and adjacent relative poses into a global reconstruction without persistent learned long-range memory. 

---

## 📣 Highlights & New Features

- **Native WebGL2 3DGS Viewer:** Interactive 3D Gaussian Splatting rasterizer built directly into the web viewer (`viewer/index.html`), supporting smooth rendering of millions of splats with point/splat toggling.
- **Closed-Form PCD to 3DGS Converter (`scripts/pcd_to_3dgs.py`):** Instantly converts dense/sparse point clouds into standard 3D Gaussian Splatting PLY format without neural re-training.
- **Real-Time 8090 Streaming Engine & 8088 Live Visualizer Bridge:** WebSocket (`/ws/viewer`) and SSE (`/api/stream/live`) broadcast channels for zero-latency point cloud and camera trajectory streaming.
- **Dynamic Object Filtering (YOLO-seg):** Real-time semantic masking and dynamic point cloud pruning.
- **Method 2 Multi-Stream Incremental Scene Fusion:** Sim(3) Pose Graph Optimization (PGO), multi-scale VGICP registration, and Statistical Outlier Removal (SOR) for multi-robot collaborative mapping.

---

## 🚀 Migration & Deployment Guide

This section provides complete instructions for deploying ABot-Recon on a fresh server or migrating an existing deployment.

### 1. System & Hardware Requirements

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **OS** | Linux (Ubuntu 20.04 / 22.04 LTS) | Ubuntu 22.04 LTS x86_64 |
| **GPU** | NVIDIA GPU with CUDA compute capability $\ge 8.0$ (RTX 3090, 4090, A10, A100, H100) | RTX 4090 (24GB) or A100 (40GB/80GB) |
| **VRAM** | 8 GB (Single stream offline inference) | 16 GB ~ 24 GB (Concurrent 8090 streaming + YOLO-seg + 8088 WebGL2 3DGS visualizer) |
| **CPU / RAM** | 8 Cores, 16 GB System RAM | 16+ Cores, 32 GB ~ 64 GB System RAM |
| **Storage** | $\ge 25\text{ GB}$ free SSD space (Model weights + cache + outputs) | NVMe SSD $\ge 100\text{ GB}$ |
| **Python & CUDA** | Python 3.10 ~ 3.11, CUDA 12.1 | Python 3.11, CUDA 12.1, PyTorch 2.5.1 |

---

### 2. Environment Setup & Dependency Installation

#### Step 2.1: Create Conda Virtual Environment
```bash
conda create -n abot-recon python=3.11 -y
conda activate abot-recon
```

#### Step 2.2: Install PyTorch with CUDA 12.1
```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

#### Step 2.3: Install All Project Dependencies
You can install dependencies via [requirements.txt](requirements.txt) or directly via editable package installation:

```bash
# Option A: Install from requirements.txt (Recommended for server deployment)
pip install -r requirements.txt

# Option B: Install editable package with all optional modules
pip install -e ".[all]"
```

#### Step 2.4: Build Acceleration Extensions (FlashInfer & cuRoPE)
ABot-Recon utilizes paged KV-cache operators from FlashInfer and custom CUDA kernels for rotary position encoding (cuRoPE):

```bash
# 1. FlashInfer paged KV-cache
pip install flashinfer-python
flashinfer show-config

# 2. Compile cuRoPE CUDA extension
cd abot_recon/modeling/pi3/models/curope
pip install ninja
python setup.py build_ext --inplace
cd -
```

---

### 3. Model Weights & Asset Deployment

The system requires several pre-trained model weights. Place them in the following directory layout:

```text
ABot-Recon/
├── checkpoints/
│   ├── abot_recon.safetensors         # [Required] Core ABot-Recon model (~4.0 GB)
│   └── loop/                          # [Optional] Loop closure retrieval weights
│       ├── dino_salad.ckpt            # (~352 MB)
│       └── dinov2_vitb14_pretrain.pth # (~346 MB)
├── yolo11m-seg.pt                     # [Optional] Dynamic object filter model (~45 MB)
└── yolo11n-seg.pt                     # [Optional] Lightweight dynamic filter model (~6 MB)
```

#### Downloading Checkpoints

* **ABot-Recon Core Model (~4.0 GB)**:
  - From **Hugging Face**: [acvlab/ABot-Recon](https://huggingface.co/acvlab/ABot-Recon)
  - From **ModelScope**: [amap_cvlab/ABot-Recon](https://modelscope.cn/models/amap_cvlab/ABot-Recon)

  ```bash
  mkdir -p checkpoints
  
  # Download using huggingface-cli
  huggingface-cli download acvlab/ABot-Recon abot_recon.safetensors --local-dir checkpoints --local-dir-use-symlinks False
  
  # Or download from ModelScope (fast in Mainland China)
  python -c "from modelscope import snapshot_download; snapshot_download('amap_cvlab/ABot-Recon', local_dir='checkpoints')"
  ```

* **Loop Closure Assets (Optional, ~700 MB)**:
  ```bash
  python scripts/download_loop_assets.py --output-dir checkpoints/loop
  ```

* **YOLO-seg Dynamic Removal Model (Optional, ~45 MB)**:
  ```bash
  # Automatically downloaded on first use, or pre-download manually:
  python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
  ```

> [!TIP]
> **Proxy Configuration for Large File Downloads**: If deploying in an environment requiring an outbound proxy to reach Hugging Face or GitHub, configure your proxy environment (e.g. using `proxy_download` or setting `export HTTP_PROXY=... HTTPS_PROXY=...`).

---

### 4. Dual-Service Architecture & One-Click Management

ABot-Recon features a dual-service architecture designed for both online real-time reconstruction and high-performance 3D visualization:

```
┌────────────────────────────────────────────────────────┐
│               ABot-Recon Dual-Service Architecture     │
├─────────────────────────┬──────────────────────────────┤
│  Port 8088: Web Viewer  │  Port 8090: Streaming API    │
│  - Native WebGL2 3DGS   │  - Real-time Multi-Session   │
│  - Point Cloud Render   │  - WebSocket / SSE Stream    │
│  - Camera Trajectory    │  - YOLO Dynamic Filter       │
│  - Relays to 8090       │  - Incremental Scene Fusion  │
└─────────────────────────┴──────────────────────────────┘
```

The unified management script [`manage_services.sh`](manage_services.sh) handles start, stop, restart, status check, and log monitoring.

#### Service Management Commands

```bash
# 1. Start all services (Port 8088 and Port 8090)
./manage_services.sh start all

# 2. Check service status, ports, and PIDs
./manage_services.sh status

# 3. View real-time service logs
./manage_services.sh log 8088    # 3D Viewer logs
./manage_services.sh log 8090    # Real-time Streaming API logs

# 4. Restart or stop services
./manage_services.sh restart all
./manage_services.sh stop all
```

#### Firewall & Reverse Proxy (Nginx) Configuration

Ensure ports **8088** and **8090** are open in your server firewall:
```bash
sudo ufw allow 8088/tcp
sudo ufw allow 8090/tcp
```

When placing behind an **Nginx** reverse proxy, ensure WebSocket upgrade headers are properly forwarded:
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

### 5. Real-Time Streaming & Live Camera Mapping

You can push live camera streams (USB camera, IP/RTSP camera, or video file) to Port 8090 for online 3D mapping and observe the point cloud growth in real-time on Port 8088:

```bash
# Push live USB camera (device 0)
python scripts/stream_camera_to_8090.py --source 0 --session-id office_cam --fps 12

# Push RTSP network camera stream
python scripts/stream_camera_to_8090.py --source "rtsp://admin:password@192.168.1.100:554/stream" --session-id rtsp_robot

# Push local video at real-time recording speed
python scripts/stream_camera_to_8090.py --source data/mine/video1.mp4 --session-id video1_stream --fps 15
```

**Live Web Observation**: Open `http://<SERVER_IP>:8088` in your browser, switch to the **"8090 连接的 Stream 视频流"** tab, and select your active session. The viewer will render camera frustums and incremental point clouds in real time.

---

### 6. Point Cloud to 3DGS Gaussian Splatting & WebGL2 Rendering

Convert any reconstructed point cloud into a standard 3D Gaussian Splatting PLY model using closed-form covariance and spherical harmonics estimation:

```bash
python scripts/pcd_to_3dgs.py \
  --input outputs/demo/reconstruction.ply \
  --output outputs/demo/reconstruction_3dgs.ply \
  --knn 16 \
  --opacity 0.85
```

- **Interactive WebGL2 Splat Viewer**: Open `http://<SERVER_IP>:8088`, load the generated 3DGS PLY, and toggle between **Points** and **3DGS Splats** modes to inspect rasterized Gaussian ellipsoids with full lighting and rotation control.

---

### 7. Multi-Stream Scene Fusion Pipeline (Method 2)

Merge multiple video trajectories and point clouds into a globally consistent coordinate system with Sim(3) Pose Graph Optimization and multi-scale VGICP:

```bash
python scripts/match_and_fuse_method2.py \
  --models outputs/stream1/reconstruction.ply outputs/stream2/reconstruction.ply \
  --output-dir outputs/fused_scene \
  --pgo \
  --sor
```

---

## 💻 Python API & Offline Inference

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

trajectory = result.camera_poses      # (N, 4, 4) world-to-camera or camera-to-world
relative_poses = result.relative_poses # (N-1, 4, 4)
local_points = result.local_points     # (N, H, W, 3)
confidence = result.confidence         # (N, H, W)
```

Export dense colored point clouds to PLY:
```bash
python scripts/export_reconstruction_ply.py \
  --poses outputs/demo/camera_poses.npy \
  --points outputs/demo/local_points.pt \
  --colors outputs/demo/colors.pt \
  --output outputs/demo/reconstruction.ply \
  --bev-output outputs/demo/trajectory_bev.png
```

---

## 🛠️ Migration & Deployment Troubleshooting Checklist

| Issue | Cause | Solution |
|---|---|---|
| **CUDA Out of Memory (OOM)** | Video frames too long or resolution too high for GPU VRAM | 1. Add `--dense-stride 2` or `--point-stride 4`<br>2. Run without dynamic filter: omit `--dynamic-filter`<br>3. Lower input stream resolution (e.g. 504x280) |
| **FlashInfer ImportError / ABI mismatch** | FlashInfer compiled against a different PyTorch / CUDA version | Set `--attention-backend sdpa` to fallback to PyTorch native scaled dot-product attention |
| **cuRoPE compilation fails** | Missing `ninja` or incompatible GCC version | Run `pip install ninja` and verify `gcc --version` ($\ge 9.0$). cuRoPE is optional; model falls back to PyTorch RoPE automatically |
| **Port 8088 / 8090 already in use** | Stray background process holding the port | Run `./manage_services.sh stop all` or identify process with `lsof -i :8088` and `kill -9 <PID>` |
| **WebSocket disconnects immediately** | Nginx or proxy dropping WebSocket upgrade headers | Add `proxy_set_header Upgrade $http_upgrade;` and `proxy_set_header Connection "upgrade";` to Nginx config |
| **Model weights download timeout** | Network latency to Hugging Face | Use ModelScope mirror `snapshot_download('amap_cvlab/ABot-Recon')` or configure proxy environment |

---

## 🧪 Tests

```bash
# Run unit tests
pytest -q

# Run cuRoPE parity test
ABOT_RECON_REQUIRE_CUROPE=1 pytest -q tests/test_curope_parity.py

# Run real checkpoint integration test
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_IMAGE_DIR=examples/images \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_checkpoint.py
```

---

## 📖 Citation

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

## 📄 License & Acknowledgements

Source code is released under the [Apache License 2.0](LICENSE). Model weights are governed by [MODEL_LICENSE.md](MODEL_LICENSE.md), and third-party components are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Before using the model, please review the [Model Usage Guidelines](MODEL_USAGE_GUIDELINES.md).

ABot-Recon builds on Pi3 and draws inspiration from CroCo, DUSt3R, DINOv2, SALAD, FlashInfer, LingBot-Map, HorizonStream, and LongStream. We thank their authors and contributors.

Special thanks to Zengye Ge, Hongyu Pan, Zhongxu Sun, Bentao Wang, Yuting Xu, Tianjian Ouyang, Haoming Yu, Chuzi Chen, and Zhiyang Zhang for their valuable support.

---

## 🌐 Other Works from Our Group

- [ABot-Earth](https://abot-earth.amap.com/)
