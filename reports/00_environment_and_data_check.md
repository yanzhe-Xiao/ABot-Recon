# 环境与数据检查报告

检查日期：2026-09-08。本报告只覆盖准备阶段，没有启动视频重建或参数 sweep。

## Git 状态

- 当前分支：`parameter-eval`
- 当前提交：`195cb9240ffc6300e008d2b70e54d281dd7caf4b`（与 `upstream/main` 相同）
- `origin`：`https://github.com/yanzhe-Xiao/ABot-Recon.git`
- `upstream`：`https://github.com/amap-cvlab/ABot-Recon.git`
- 工作区 dirty：是。已有改动均被保留并带入新建的 `parameter-eval` 分支。
- 来源待确认：`THIRD_PARTY_NOTICES.md` 删除、`scripts/download_loop_assets.py` 权限位变化、9 月 1 日生成的 benchmark/visualization 脚本。
- `reports/tunable_parameters.md` 是此前参数源码分析；视频、场景 SVG 和数据说明是本次输入。
- 已为视频、权重、点云、数组等扩展名补充 `.gitignore`，避免约 746 MiB 原视频被误提交；场景 SVG 和 Markdown 仍可跟踪。

## 运行环境

| 项目 | 结果 |
|---|---|
| Conda | `abot`，Python 3.11.16 |
| PyTorch | 2.5.1+cu124，torch CUDA 12.4 |
| CUDA | 可用；驱动 570.133.07（支持 CUDA 12.8），nvcc 12.6 |
| GPU | 6 × RTX 3090；检查时每卡 15 MiB、0% utilization |
| checkpoint | `/data/user24302666/abot_navigation_comparison/20260901_224942_inventory/caches/abot_model/abot_recon.safetensors`，SHA256 `ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e` |
| strict load | 通过；3.30 s，峰值约 3862.4 MiB |
| attention | `auto → sdpa`；FlashInfer 未安装 |
| cuRoPE | CUDA 扩展未编译，当前使用 PyTorch fallback |
| loop assets | 缺少 `dino_salad.ckpt` 与 `dinov2_vitb14_pretrain.pth` |
| ffmpeg/ffprobe | 4.2.7 / 4.2.7 |
| 测试 | 49 passed, 7 skipped |

已按用户授权在 `abot` 环境安装 Open3D、SciPy、scikit-learn、FAISS、Matplotlib、OpenCV、Pandas、ImageIO、psutil、plyfile、PyPose 与 pytest。PyTorch/CUDA 版本未变化。FlashInfer 只做了 dry-run；最新包会引入一组 CUDA 13 时代的辅助依赖，本轮未安装。

## 视频检查

所有原视频均完整逐帧解码通过。均为 DJI OsmoPocket3、1920×1080、29.97 FPS、HEVC Main 10、`yuv420p10le`，无 rotation metadata，按帧时间戳未检测到 VFR。码率存在正常的小幅差异，因此“同相机模式”结论基于可见编码元数据。

| ID | 文件名 | 时长(s) | 帧数 | 分辨率 | FPS | 编码 | 像素格式 | VFR | 解码 |
|---|---|---:|---:|---|---:|---|---|---|---|
| 01 | 01.MP4 | 40.541 | 1215 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 02 | 02.MP4 | 36.403 | 1091 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 03 | 03.MP4 | 23.223 | 696 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 04 | 04.MP4 | 30.497 | 914 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 05 | 05.MP4 | 64.131 | 1922 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 06 | 06.MP4 | 30.530 | 915 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 07 | 07.MP4 | 38.839 | 1164 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |
| 08 | 08.MP4 | 24.691 | 740 | 1920×1080 | 29.97003 | hevc Main 10 | yuv420p10le | 否 | 通过 |

### 06_clean

- 规范文件：`/data/user24302666/abot_video_eval/source_info/derived_videos/06_clean.MP4`
- 855 帧，28.5285 s；相对原始 30.5305 s 去掉 2.0020 s。
- 使用 stream copy，未重新编码；主视频保持 HEVC Main 10 / 10-bit。
- 完整解码通过，SHA256：`b4b00615ee427d5d26a921d88eb4f4af9667618684750ccd253fd135d9983433`。
- 首次 `-map 0` 因 DJI timecode/data track 无法封装而失败，日志已保留；随后显式处理主视频/音频流。
- 856 帧候选比目标长 31 ms，已保留为带 attempt 后缀的审计文件；855 帧规范版本只偏离目标裁剪点约 2 ms。

## 现有代码能力

- `scripts/export_reconstruction_ply.py` 已能从 local points + poses 生成二进制 RGB PLY，并支持 confidence、point/frame stride 和最大点数。可复用。
- 未跟踪的 `scripts/benchmark_video_reconstruction.py` 已有 SDPA/paged、dtype、dense stride、推理计时、峰值显存和核心 tensor 保存，但只接受图像目录，且缺少 stride、完整 Git/config metadata、总耗时、质量指标、CSV、失败状态、loop 配置和统一后处理编排。
- 未跟踪的 `scripts/visualize_reconstruction_video.py` 能生成输入图像与累计点云的组合视频，但仅生成一种 MP4，且每个实验独立 auto-fit；不满足统一 bounding box/camera、overview.mp4 与 accumulation.mp4 两类输出要求。仓库根目录还有一个较旧重复版本。
- 仓库没有跨视频 registration、FPFH/RANSAC/ICP baseline、自动 metrics、confidence 离线 sweep、GPU 调度、统一 CSV/图表/报告生成器。
- 官方 demo/CLI 以图像目录为输入，不直接处理视频。
- smoke/baseline（loop=false）不需要修改模型核心；需要新增评测编排、指标、统一渲染和视频抽帧代码。
- 后续 rotation sweep 需要为当前硬编码的 `rot_correction_max_deg=2.0` 增加可审计入口；loop 参数实验需要让高层 API 接收完整 `LoopClosureConfig`。这些应做小范围源代码扩展并配测试。

## Smoke test 准备

- 输入：`06_clean` 前 120 帧，逐帧无损 PNG，1920×1080，共约 886.7 MiB，清单含逐文件 SHA256。
- 配置：`/data/user24302666/abot_video_eval/smoke_test/experiment_config.json`。
- baseline 参数已写入配置：stride=1、dense_stride=1、confidence=0、280×504、bf16、auto→sdpa、window=12、rotation kernel=10/max=2°、loop=false。
- 当前状态：输入和配置已准备，尚未推理。现有 benchmark 还不能生成要求中的两个 MP4、完整 metrics.json、video_results.csv 和可复现状态记录，因此不应直接把它作为正式 smoke pipeline。

## 阻塞与下一步

1. 先实现并测试评测 orchestrator、质量指标、两种统一 MP4、CSV/status/log/config 原子写入。
2. 使用准备好的 120 帧在 GPU 0 做 smoke；全部 20 项验收通过后，再跑一个完整 baseline 视频。
3. Loop closure 阶段前下载并校验两个 loop assets；本轮 loop=false 不受影响。
4. paged backend 当前不可测；若要纳入第一轮，需单独评估与现有 Torch 2.5/CUDA 12.4 兼容的 FlashInfer 版本，不能直接安装最新依赖集。
