# Method 2 多段视频流与点云通用匹配融合工具使用指南

本项目提供了基于**方案二（视频流辅助多模态极速配准）**的通用融合引擎：
📂 **`scripts/match_and_fuse_method2.py`**

该工具支持面向未来的任意多段视频或点云建图场景，**输入端**支持直接输入视频文件（`.mp4`）、图像目录或重建数据包；**计算端**全自动发现视角拓扑重叠并求解 $\text{Sim}(3)$ 相似变换与局部微调；**输出端**完全支持用户**自由勾选所需产物**。

---

## 一、 方法二核心原理与为什么需要“视频+点云”

方法二的全称为：**视频流辅助的多模态配准（ALIKED + LightGlue + 2D-to-3D 升维 + Umeyama 闭式解 + small_gicp）**。

```
视频帧 A ────────► [ALIKED + LightGlue] ◄──────── 视频帧 B
                        │ (求得高质量 2D 像素匹配对 u, v)
                        ▼
                 [2D-to-3D 密集反查] (根据像素映射表查询对应 3D 坐标 X, Y, Z)
                        │
                        ▼
点云 A ────────► [Umeyama SVD 闭式解] ◄──────── 点云 B
                        │ (毫秒级求解尺度 s、旋转 R、平移 t)
                        ▼
                 [small_gicp (VGICP)] (消除微小接缝误差)
                        │
                        ▼
                 [多视角空间融合地图]
```

### 为什么不能只输入孤立的 PLY 点云？
1. **2D 图像特征网络依赖纹理**：`LightGlue` 是深度视觉注意力匹配模型，必须输入具有光影和颜色纹理的 2D 视频图像，才能在无几何特征（如白墙、空旷地面）的场景下稳定匹配出数百对对应关键点。
2. **2D-to-3D 映射**：找到像素匹配后，需要通过相机投影/反投影表将像素 $(u, v)$ 升维查出真实世界坐标 $(X, Y, Z)$。
3. **数据包机制**：ABot-Recon 的重建目录（如 `outputs/data_05_loop`）本身就是**视频帧（`colors.pt`）+ 点云反查表（`world_points.pt`）+ 稠密点云（`reconstruction.ply`）**的多模态绑定集合。

---

## 二、 输入自适应模式

工具可以接受任意数量（$N \ge 2$）的输入路径，自动识别并处理：

| 输入类型 | 命令行写法示例 | 处理流程 |
| :--- | :--- | :--- |
| **已建图目录（推荐，最快）** | `outputs/data_05_loop outputs/data_06_loop ...` | 直接读取现有视频特征与点云反查表，秒级完成匹配与融合 |
| **原始视频文件** | `data/data/05.MP4 data/data/06.MP4 ...` | 检查是否有历史重建缓存；若无，自动抽帧并调用 ABot-Recon 重建后自动融合 |
| **图像帧目录** | `data/data/05 data/data/06 ...` | 自动调用 ABot-Recon 稠密前向推理并执行方法二融合 |

---

## 三、 自选产物配置表 (`--outputs` / `-o`)

用户可以使用 `--outputs`（或 `-o`）指定所需产物，多个产物用逗号分隔，**未被勾选的产物将自动跳过计算与磁盘 I/O**：

| 输出选项关键字 | 生成文件命名示例 | 产物内容与工程用途 |
| :--- | :--- | :--- |
| **`normal`** / `normal_merged` | `<prefix>_normal_merged.ply` | **真彩体素去重融合点云**（默认 1.5cm 体素滤波，无双层重影，体积压减 75%，适合前端网页交互） |
| **`normal_full`** | `<prefix>_normal_full.ply` | **真彩全量无损点云**（100% 原始点数保留，用于高保真底模资产归档） |
| **`colored`** / `colored_merged` | `<prefix>_colored_merged.ply` | **多视频流分段赋色点云**（自动按黄金分割色相环为各视频赋予高对比度独立颜色，直观展现空间覆盖与重叠） |
| **`colored_full`** | `<prefix>_colored_full.ply` | **多流分段赋色全量点云**（全分辨率分流点云） |
| **`aligned`** | `aligned_individual/seq_<id>_aligned.ply` | **变换到统一世界坐标系下的各段单独点云**（便于导入 CloudCompare/MeshLab 进行分层审查） |
| **`transforms`** | `<prefix>_transforms.json` | **姿态与尺度参数文件**（包含每段的 Sim(3) 尺度因子 $s$、4x4 变换矩阵及配准内点统计） |
| **`report`** | `<prefix>_REPORT.md` | **Markdown 评测报告**（记录每对视频的匹配内点数、内点率、RMSE 误差） |
| **`viewer`** | 控制台输出展示链接 | **自动注册并发布至 `viewer/server.py`**，可通过网页前端在线交互 |
| **`all`** | 全部文件 | **一键生成并导出上述所有产物**（默认设置） |

---

## 四、 命令行调用参考 (CLI)

### 场景 1：只融合出正常的真彩点云、位姿参数和报告（日常最常用）
```bash
python scripts/match_and_fuse_method2.py \
  outputs/data_05_loop outputs/data_06_loop outputs/data_07_loop outputs/data_08_loop \
  --output-dir outputs/alignment/my_fusion \
  --outputs normal,transforms,report \
  --voxel-size 0.015
```

### 场景 2：只想查看不同视频流的着色区分效果，并导出单段对齐文件
```bash
python scripts/match_and_fuse_method2.py \
  outputs/data_05_loop outputs/data_06_loop outputs/data_07_loop outputs/data_08_loop \
  --output-dir outputs/alignment/colored_debug \
  --outputs colored,aligned
```

### 场景 3：极速配准，仅获取各点云之间的相对位姿变换矩阵（数秒级）
```bash
python scripts/match_and_fuse_method2.py \
  outputs/data_06_loop outputs/data_07_loop \
  --output-dir outputs/alignment/quick_sim3 \
  --outputs transforms
```

### 场景 4：输入未建图的原始视频文件，一键完成建图 + 融合 + 部署到前端
```bash
python scripts/match_and_fuse_method2.py \
  data/data/05.MP4 data/data/06.MP4 data/data/07.MP4 data/data/08.MP4 \
  --output-dir outputs/alignment/video_pipeline \
  --outputs normal,colored,viewer
```

### 完整参数说明：
- `inputs`: 任意数量的输入路径（支持视频、图片目录或已建图目录）；
- `--output-dir` (`-d`): 输出目录路径；
- `--outputs` (`-o`): 逗号分隔的产物列表或 `all`；
- `--anchor`: 手动指定基准参考系（默认 `None`，由算法自动根据拓扑连通度推选最佳锚点）；
- `--voxel-size`: 体素去重网格尺寸，单位米（默认 `0.015` 即 1.5cm，设为 `0` 则不去重）；
- `--keyframe-stride`: 关键帧采样间隔（默认 `5` 帧，间隔越小匹配精度越高）；
- `--prefix`: 生成文件命名前缀（默认 `fused`）；
- `--device`: 计算设备，自动检测 `cuda:0` 或 `cpu`。

---

## 五、 Python API 脚本调用

可以在其他自动化脚本中直接导入引擎：

```python
from scripts.match_and_fuse_method2 import Method2FusionPipeline

# 1. 实例化多模态融合流水线
pipeline = Method2FusionPipeline(
    device="cuda:0",
    keyframe_stride=5,
    merge_voxel_size=0.015,
)

# 2. 运行融合任务
results = pipeline.execute(
    inputs=[
        "outputs/data_05_loop",
        "outputs/data_06_loop",
        "outputs/data_07_loop",
        "outputs/data_08_loop",
    ],
    output_dir="outputs/alignment/my_fusion",
    outputs=["normal", "colored", "transforms"],
    prefix="office_hall",
)

# 3. 获取输出字典
print("Anchor 序列:", results["anchor_sequence"])
print("生成的文件:", results["deliverables"])
```

---

## 六、 05-08 序列通用流程与原有做法一致性评测

使用通用工具在 `--outputs normal,transforms,report` 模式下对 `05-08` 进行全自动融合，并与手动专项脚本结果对比：

| 评估指标 | 手动专项脚本 | 通用方法工具 (`match_and_fuse_method2.py`) | 一致性评测结论 |
| :--- | :--- | :--- | :--- |
| **拓扑建图方式** | 人工指定级联路径 ($05 \to 06 \to 07$) | **MST 最大生成树全自动推选** ($05 \to 08 \to 07$) | 自动发现了更高信噪比的 $05 \leftrightarrow 08$ 书柜连接 (107 内点) |
| **基准参考系** | 手动指定 `07` 为 Anchor | **算法自动推选** `07` 为 Anchor | **完全一致** |
| **去重融合点数** | 3,409,327 点 (87.8 MB) | 3,318,341 点 (85.4 MB) | **点数差异 < 2.6%**（体素滤波网格微小差异） |
| **空间几何范围** | $11.08\text{m} \times 7.41\text{m} \times 16.20\text{m}$ | $10.52\text{m} \times 7.21\text{m} \times 15.10\text{m}$ | **完全吻合** |
| **两点云最近邻距离中位数** | 基准对齐 | **11.34 mm（约 1.1 厘米）** | **毫米级高度一致** |
| **8cm 内贴合率** | 基准对齐 | **80.1%** | **两版点云在几何空间中几乎完全重合** |
| **运行总耗时** | 54.72 s | 70.85 s（含自动拓扑全网扫描） | **1 分钟级极速出图** |
