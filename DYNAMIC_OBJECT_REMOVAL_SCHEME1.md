# 动态场景单目 3D 点云重建：动态物体滤除方案一（2D 语义实例先验掩码）技术设计与实现指南

## 一、方案概述与背景痛点

在**单目无深度传感器、无先验位姿（Monocular, Depth-less, Pose-free）**的自然场景 3D 点云重建中，场景中存在移动物体（如走动的人员、推车、行驶车辆等）会引发两大核心灾难：

1. **几何拉丝与鬼影重叠（Ghosting & Trailing Artifacts）**：
   单目神经网络（如 ABot-Recon / PI3 / DUSt3R 族系）在每一帧都会预测当前局部坐标系下的点图 $P_i \in \mathbb{R}^{H \times W \times 3}$。当物体移动时，不同时间戳下物体处于空间不同位置，经过全局相机位姿 $T_{w \leftarrow c_i}$ 变换到世界坐标系后叠加，会在运动轨迹上生成连续的**多重悬空残影和拉丝碎点**。
2. **相机位姿跟踪漂移（Pose Estimation Drift）**：
   流式建图依赖视觉特征的刚体一致性。如果画面中大面积区域存在独立运动，位姿估计器会将物体运动误判为相机位姿变化，导致轨迹失真甚至系统发散。

**方案一（基于 2D 语义实例分割先验的动态掩码过滤法）** 是一种高鲁棒、低延迟、工程即插即用的轻量级解决方案。它利用成熟的通用目标检测与实例分割模型（如 YOLOv11-seg / YOLOv8-seg / MobileSAM），在视频帧进入三维建图管道时实时识别潜在动态实体，生成高精度像素级掩膜，从源头上阻断动态点云的生成与累积。

---

## 二、数学建模与张量对齐机理

### 1. 像素与局部点图的天然对齐
ABot-Recon 采用 Transformer 逐 Patch/逐像素回归的方式，单帧推理输出的局部 3D 点图为：
$$P_i^{\text{local}} \in \mathbb{R}^{H \times W \times 3}$$
置信度图为：
$$C_i \in \mathbb{R}^{H \times W}$$
其中图像坐标 $(u, v)$ 与点图索引 $(v, u, :)$ **严格空间共形、一一对应**。

### 2. 动态掩码构建与形态学膨胀
针对第 $i$ 帧图像 $I_i$，通过语义实例分割模型预测所有动态目标类别集合 $\mathcal{C}_{\text{dynamic}} = \{\text{person, bicycle, car, motorcycle, bus, truck, dog, cat, ...}\}$ 的掩模：
$$M_{\text{dynamic}}^{(i)} = \bigcup_{k \in \mathcal{K}_{\text{dyn}}} M_k^{(i)} \in \{0, 1\}^{H \times W}$$
静态掩模定义为补集：
$$M_{\text{static}}^{(i)} = 1 - M_{\text{dynamic}}^{(i)}$$

为了彻底消除物体运动边缘由于分割轻微内缩引起的“边界毛刺点云”，引入形态学膨胀操作（Kernel Size $K = 5 \sim 9$ px）：
$$\widetilde{M}_{\text{dynamic}}^{(i)} = M_{\text{dynamic}}^{(i)} \oplus \mathcal{B}_K$$
$$\widetilde{M}_{\text{static}}^{(i)} = 1 - \widetilde{M}_{\text{dynamic}}^{(i)}$$

### 3. 逐点几何置零与置信度屏蔽
将静态掩模作用于置信度图与局部点图：
$$\widetilde{C}_i(u, v) = C_i(u, v) \cdot \widetilde{M}_{\text{static}}^{(i)}(u, v)$$
$$\widetilde{P}_i^{\text{local}}(u, v) = \begin{cases} P_i^{\text{local}}(u, v), & \text{if } \widetilde{M}_{\text{static}}^{(i)}(u, v) = 1 \\ \text{NaN}, & \text{if } \widetilde{M}_{\text{static}}^{(i)}(u, v) = 0 \end{cases}$$

经世界变换 $T_{w \leftarrow c_i}$ 后，仅有有效静态点被变换并写入全局点云，动态物体在 3D 空间完全被忽略不计。

```
【输入单目视频帧 I_i】
         │
         ├───► 【YOLOv11-seg 实例分割】 ──► 【类别过滤 (person等)】 ──► 【动态掩码 M_dyn 膨胀】
         │                                                                   │
         ▼                                                                   ▼
【ABot-Recon 流式网络】 ──► 【局部点图 P_i & 置信度 C_i】 ──► 【掩膜置零 / 剔除 NaN 点】
         │                                                                   │
         ▼                                                                   ▼
【相机位姿 T_{w<-c_i}】 ──────────────────────────────────────────► 【纯静态无鬼影全局 3D 点云】
```

---

## 三、方案一的核心技术优势

1. **零改动网络架构，极速集成**：
   无需重新训练或微调 ABot-Recon 权重，通过外挂轻量分割模块（如 YOLOv8n-seg / YOLO11n-seg，单帧推理仅 4~8 ms），以流水线前处理/后处理方式无缝集成。
2. **100% 阻断常见动态物体**：
   对室内走廊中的行人、会议室中的人员、园区道路上的车辆等高频动态目标具有极高的召回率和清晰的边界剥离能力。
3. **显存友好，计算开销可忽略**：
   YOLO-seg 仅占用数百兆显存，可与 ABot-Recon 在同一张 GPU（甚至纯 CPU）上无冲突并发执行。

---

## 四、代码实现与工程接入规范

### 1. 核心分割器封装 (`dynamic_mask.py`)
```python
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class DynamicObjectMasker:
    def __init__(self, model_name="yolo11n-seg.pt", dynamic_classes=None, device="cuda"):
        self.model = YOLO(model_name)
        # 默认动态目标：人(0), 自行车(1), 汽车(2), 摩托车(3), 公交车(5), 卡车(7), 猫(15), 狗(16), 椅子(56-可选)
        self.dynamic_classes = dynamic_classes or [0, 1, 2, 3, 5, 7, 15, 16]
        self.device = device

    def predict_mask(self, image_path_or_rgb, conf_thresh=0.25, dilate_kernel=7):
        """返回 H x W 的布尔掩码：True 表示静态，False 表示动态物体"""
        results = self.model.predict(
            image_path_or_rgb,
            classes=self.dynamic_classes,
            conf=conf_thresh,
            device=self.device,
            verbose=False,
        )
        if isinstance(image_path_or_rgb, str):
            img = cv2.imread(image_path_or_rgb)
            h, w = img.shape[:2]
        else:
            h, w = image_path_or_rgb.shape[:2]

        dynamic_mask = np.zeros((h, w), dtype=np.uint8)
        if len(results) > 0 and results[0].masks is not None:
            for mask_data in results[0].masks.data:
                m = mask_data.cpu().numpy().astype(np.uint8)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                dynamic_mask = np.bitwise_or(dynamic_mask, m)

        # 形态学膨胀，消除边缘毛刺
        if dilate_kernel > 0 and np.any(dynamic_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
            dynamic_mask = cv2.dilate(dynamic_mask, kernel)

        static_mask = dynamic_mask == 0
        return static_mask, dynamic_mask
```

### 2. 与 ABot-Recon 3D 点云流式建图对接
在点云写入或变换至世界坐标系时：
```python
# computed_local_points: [H, W, 3], confidence: [H, W]
static_mask_torch = torch.from_numpy(static_mask).to(device=computed_local_points.device)
invalid_mask = (~static_mask_torch) | (confidence < confidence_threshold)

# 动态区域与低置信度区域全部置为 NaN
valid_points = computed_local_points[~invalid_mask] # 仅保留纯静态背景 3D 点
```

---

## 五、验证与评估指标体系

在含动态人员的测试视频（如 `/home/data/xyz/ABot-Recon/data/玻璃房/玻璃房1.mp4`）中，评测如下指标：
1. **动态区域点云滤除率（Dynamic Point Rejection Ratio）**：统计被行人包围盒内 3D 点被成功屏蔽的比例（目标 $> 98\%$）；
2. **静态背景保留度（Static Background Retention）**：评估沙发、茶几、玻璃窗、地面等背景结构点云的完整度；
3. **单帧额外处理耗时（Overhead Latency）**：对比开启掩码前后的 FPS 变化（目标额外延迟 $< 10\text{ ms}$）。
