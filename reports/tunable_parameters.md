# ABot-Recon 可调参数源码审计

## 1. 审计范围与结论口径

本报告审计当前 `main` 分支（提交 `195cb92`）的公开推理代码，并只读参考了同仓库 `origin/eval` 分支中的评测适配器。检查范围包括 CLI、`InferenceConfig`、模型构造、相邻帧 Pose head、temporal rotation refinement、流式 attention/KV cache、RoPE、点图与 confidence 输出、loop closure、SALAD retrieval、GPU pose graph、图像预处理、checkpoint strict load，以及仓库内 PLY 导出后处理。

当前公开仓库没有训练代码和训练 recipe；README 也明确标注训练代码尚未发布。因此，凡涉及训练时采样分布的判断只能依据：发布构造器中的固定值、源码注释、公开 checkpoint 的真实 tensor key/shape，以及 `origin/eval` 的固定评测设置。无法从训练 recipe 直接证明的地方标记为“待验证”。

分类定义：

- **A 安全的纯推理参数**：不改变权重结构，适合直接做推理实验。
- **B 谨慎的推理参数**：代码或 strict load 可以工作，但会改变训练时 context、数值范围或输入分布。
- **C 模型结构/checkpoint 绑定参数**：改变后会造成 missing/unexpected key、size mismatch，或改变主体架构。
- **D 纯输出/存储参数**：只控制返回、保存或离线抽样；少数参数可通过跳过输出 head 间接提速。
- **E Loop Closure 专用参数**：只在回环后端生效。
- **F 训练专用参数**：推理阶段没有正常调参意义。

建议等级严格使用：**强烈建议测试、建议测试、谨慎测试、不建议修改、不可修改**。

## 2. 最重要的源码结论

1. **发布模型的 local context 确实固定为 12 帧。** `InferenceConfig.__post_init__` 直接拒绝非 12；`config.json`、`model.py` 和 `origin/eval:configs/model/default.yaml` 也均为 12。实际 attention 保留当前帧加前 11 帧的完整 KV。window 数值本身不产生 checkpoint tensor，底层也有 `set_local_window_frames()`，所以它不是 tensor shape 绑定；但公开 API 明确禁止修改，而且源码说明训练 runtime 从 `train.local_window_sample_values` 设置它。训练配置未发布，非 12 是否在训练中出现过无法确认。结论为 **B，公开接口不可调，非 12 不建议修改**，而不是 C。
2. **`rot_correction_kernel=10` 与 checkpoint 强绑定。** 它决定 `age_embed.weight [10,512]`、`conv.weight [512,1,10]`、`gate.weight [512,1,10]`。改成 8 的最小 strict-load 测试在这三处真实发生 size mismatch。结论为 **C/不可修改，除非改权重并重新训练或专门迁移权重后微调**。
3. **`rot_correction_max_deg=2.0` 不改变权重 shape。** 它只令最终残差为 `max_rad * tanh(out)`；用真实 rotation-refiner state_dict 在 `max_deg=1.0` 下 strict load 成功。但这会重新标定训练所得输出，逐帧旋转又会被连续累乘，属于 **B/谨慎测试**，不是安全旋钮。
4. **`max_frames` 是运行容量参数，不是输入截断参数。** CLI 只把它传给 `paged_max_total_frames` 和 `RoPE3D.max_seq_len`，`infer_paths` 仍处理传入的全部路径。公开构造中 `num_summary_tokens=0`，因此 paged KV 没有随 `max_frames` 增长的 summary pool；主要随它增长的是非持久化 RoPE 频率表。值小于实际帧索引或空间位置索引时会越界。真实完整模型以 `max_frames=64` strict load 成功，证明它不绑定 checkpoint。
5. **`dense_stride` 不改变 Pose 输入和 Pose 结果。** 所有选中帧仍执行 encoder、主 decoder、camera decoder/head 和状态更新；非 dense 帧走 `_forward_frame_camera_only()`，跳过 point decoder、confidence decoder/head 及稠密输出收集。仓库真实 checkpoint 集成测试还检查了 sparse dense output 与 full dense output 在相同帧上的结果一致，Pose bit-exact 相同。因此它能明显减少稠密 head 计算、CPU 返回内存和文件大小，但不会按比例提升整体 FPS，也不减少 Pose 的 temporal context。
6. **`confidence_threshold` 是 sigmoid 后的纯后处理。** `confidence = sigmoid(logits)` 后执行 `confidence >= threshold`，低置信点被写成 NaN；它不反馈模型，也不改变 Pose。只要已经保存未阈值化的 `local_points.pt` 和 `confidence.pt`，可用导出脚本离线重测，无需重新推理。若推理时没有请求 confidence，但 threshold 大于 0，API 会强制计算 confidence branch。
7. **公开 Python API/CLI 目前没有暴露 loop 内部超参数。** `refine_trajectory()` 每次新建只有资产路径被覆盖的默认 `LoopClosureConfig`。`retrieval_top_k`、阈值、keyframe stride、reinfer 上限和 PGO 参数只有直接调用低层 `apply_loop_closure(..., loop_cfg=...)` 或修改适配代码才会生效。它们是可实验的 E 类参数，但不是现有 `ABotRecon.infer()` 的可调参数。
8. **开启 loop closure 时，真正增加 ABot-Recon re-inference 成本的是 `max_reinfer_candidates` 和 `loop_chunk_size`，候选数量由 threshold/top-k/NMS/max-candidates 间接控制。** 每个保留候选会对源、目标附近两段帧重新跑一次仅 Pose 推理，最多约 `2 * loop_chunk_size` 个去重帧。
9. **回环只修正轨迹，不重新预测 local point map。** 若请求 world points，API 会用 loop 后 Pose 重新变换已有 local points。因此跨视频 match/fusion 最应保留 `local_points + confidence + 相机 Pose`，并在后处理阶段统一阈值、抽样与融合；只保存 world points 会丢失最灵活的可重投影基础。
10. **自定义 `height/width` 存在输出配色陷阱。** 模型推理会使用 config 中的尺寸，但 `demo.py` 和 `abot_recon/cli.py` 保存 `colors.pt` 时再次调用无参数 `preprocess_image()`，固定产生 280×504 颜色。如果模型尺寸被改动，点图与颜色 shape 将不一致。正式实验前需要同步修正保存路径或自写调用。

## 3. 关键调用链

### 3.1 输入采样与稠密输出

```text
start/end/stride
→ demo.py:collect_images 或 abot_recon/cli.py:_collect_images
→ sorted(image paths)[start:end:stride]
→ ABotRecon.infer(paths)
→ ReleasedABotReconModel.infer_paths
→ 每条 path 都进入 iter_preprocessed 与 inference_stream_iter
→ 实际 Pose 序列及 loop 的帧编号

dense_stride
→ range(0, len(selected_paths), dense_stride)
→ ABotRecon.infer(dense_output_indices=...)
→ ABotReconNetwork.inference_stream_iter
→ dense 帧执行完整 point/conf head；其他帧执行 camera-only path
→ Pose 仍覆盖全部 selected paths，点图/confidence 只覆盖 dense indices
```

`stride` 先删掉视频帧，改变模型看到的相邻运动和时间单位；`dense_stride` 在模型仍看到全部已选帧的前提下只稀疏计算/保存 dense heads。两者不能混为一谈。

### 3.2 预处理、分辨率与点图

```text
InferenceConfig.height/width
→ ReleasedABotReconModel._frames
→ iter_preprocessed
→ preprocess_image
→ RGB 转 [0,1]
→ 宽度锁定的 bicubic resize + antialias
→ 垂直中心裁剪，或用 ImageNet mean pad_rgb 垂直填充
→ model 内部再按 image_mean/image_std 标准化
→ ViT-L/14 patch embedding + 可插值 DINO position embedding
→ decoder
→ point_head 输出 (x/z, y/z, log z)
→ clamp(log z, max=point_z_log_max=10) + exp
→ local_points=(x/z*z, y/z*z, z)
→ camera pose 变换得到 world points
```

输入高宽必须是 14 的整数倍，否则 patch embedding 断言失败。高宽不改变卷积/线性权重 shape，DINO position embedding会插值，所以 strict load 兼容；但 280×504 是明确的训练 FOV 与发布评测分布，改分辨率/宽高比属于 B 类。

### 3.3 Pose 与 rotation refinement

```text
RELATIVE_CAMERA_HEAD（model.py 固定字典）
→ Pi3(camera_pose_mode="relative_adjacent")
→ AdjacentPoseHead
→ 5 个 pose/register tokens 经 frame_descriptor
→ 相邻 descriptor pair 经 pair_mlp
→ delta_t_head + delta_q_head 得到 raw adjacent SE(3)
→ TemporalRotationRefiner：相邻 descriptor + 前后图像 token attention
→ 保留最近 rot_correction_kernel 个 motion feature
→ age embedding + depthwise conv/gate
→ max_rad * tanh(out) 得到 rotation residual
→ raw rotation 右乘 correction matrix
→ 与 previous_pose 连续累乘为全局 camera pose
```

`local_window_frames=12` 属于主 decoder 的视觉 context；`rot_correction_kernel=10` 属于 Pose head 自己的 motion-feature history，二者是独立窗口。

### 3.4 Attention、KV 与序列容量

```text
attention_backend
→ resolve_attention_backend(auto/paged/sdpa)
→ paged: FlashInfer PagedKVCacheManager
   sdpa: StreamingKVState + PyTorch scaled_dot_product_attention
→ 每个奇数 global decoder block 使用 local_window_frames
→ 当前帧 + 最多 W-1 个过去完整帧 KV

max_frames
→ rope3d_config.max_seq_len（时间、高、宽三轴共享频率表容量）
→ paged_max_total_frames
→ 仅在 num_summary_tokens>0 时决定未设上限的 summary page pool
→ 发布构造 num_summary_tokens=0，所以不会决定当前 paged KV 主窗口大小
```

`attention_backend` 更换实现而不更换 attention 权重。仓库测试覆盖 backend 解析和两者的真实 checkpoint strict load，但当前仓库没有 paged-vs-SDPA 数值 parity 测试；质量一致性应在正式实验中做小样本核对。

### 3.5 Confidence 与输出

```text
checkpoint 是否含 conf_decoder./conf_head.
→ 构造时自动 enable_confidence
→ output_confidence 或 (需要点且 threshold>0)
→ output_keys 加入 conf
→ confidence decoder/head 输出 logits
→ CPU float32 sigmoid
→ confidence_threshold 生成 mask
→ local/world points 低置信位置写 NaN
```

`output_local_points/output_world_points/output_confidence` 决定是否运行和返回相应 dense branch；它们不是改变预测函数的“质量参数”。如果三个 dense 输出都不需要，网络会走 camera-only path。

### 3.6 Loop closure

```text
loop_closure=True
→ descriptor worker 与基础流式推理并发
→ 所有输入帧生成 DINOv2-SALAD descriptor
→ keyframe_stride 对 descriptor 数组再抽样
→ FAISS inner-product top-k retrieval
→ score threshold + min separation + pair 去重 + NMS + max_candidates
→ 按 score 取前 max_reinfer_candidates
→ 每候选取源/目标各 loop_chunk_size 邻域帧
→ ABot-Recon camera-only re-inference 得到相对位姿约束
→ sparse/full pose graph + odometry edges + loop edges
→ GPU PGO
→ sparse keyframe correction 经 Slerp/线性插值扩展到全轨迹
→ 如需 world points，用新轨迹重变换原 local point maps
```

注意：公开 worker 已对**所有帧**计算 descriptor，之后才按 `keyframe_stride` 抽样，所以在当前公开路径中增大 `keyframe_stride` 不会减少 SALAD CNN 的前向次数，只减少 retrieval/候选和后续 re-inference/PGO 规模。

## 4. 参数总表

“绑定”列同时给出分类。范围是首轮合理筛选范围，不代表已有 benchmark 证明其最优。

| 参数 | 默认值 | 定义位置 | 生效模块 | 主要作用 | 对质量影响 | 对速度影响 | 对显存影响 | 是否与训练/checkpoint绑定 | 是否建议调 | 建议测试范围 |
|---|---:|---|---|---|---|---|---|---|---|---|
| `checkpoint` / `filename` / `revision` | `acvlab/ABot-Recon` / `abot_recon.safetensors` / None | CLI；`api.py:26-46`; `checkpoint.py` | 模型权重解析与strict load | 选择模型版本 | 换权重会改变全部预测，不能与超参混为一项 | 只影响下载/加载 | 权重常驻量由模型决定 | C；必须与发布构造完全匹配 | **不可修改** | 固定同一checkpoint及revision，记录SHA256 |
| `image_dir` / `output_dir` | examples / outputs | CLI/demo | 输入发现、文件写入 | 数据和结果路径 | 路径本身无质量作用；输入命名按字典序排序 | 存储设备影响I/O | 无GPU影响 | D | **不建议修改** | 每组实验使用独立输出目录，保证零填充文件名 |
| `stride` | 1 | `demo.py:32,73`; `cli.py:31,75` | 输入 path 切片 | 每隔 N 个原视频帧送一帧 | 大会增大相邻运动、降低重叠，Pose/点云连续性通常变差；小保留最大信息 | 大近似按输入帧数提速 | 总输出/CPU 内存下降；峰值 KV基本不变 | A；无 checkpoint 绑定 | **强烈建议测试** | `1,2,3`；以 1 为质量基线 |
| `start/end` | 0/None | CLI/demo | 输入切片 | 选择视频区间 | 不直接改变单帧模型；改变序列初始参考和累计漂移 | 与帧数成比例 | 与帧数/输出成比例 | A | **建议测试** | 用固定代表片段做消融，不作为质量超参 |
| `dense_stride` / `dense_output_indices` | 1 / None | CLI/demo；`api.py:62`; `network.py:1278` | point/conf dense heads | 稀疏运行和保存点图；Pose仍全帧 | 保留帧上的预测与 dense=1 一致；最终融合覆盖率下降 | 大会跳过多数 point/conf decoder，提速但核心 Pose path不变 | GPU临时 head开销和CPU输出大幅下降 | A | **强烈建议测试** | `1,2,4,8`；质量优先建议 `1–2` |
| `height` | 280 | `config.py:15`; `preprocessing.py:61` | 预处理、token数、所有视觉模块 | 输出点图高度与垂直 FOV | 非训练尺寸可能改善细节或破坏标定/FOV分布，待验证 | token数上升显著变慢 | 近似随 token/KV 数上升 | B；strict compatible，训练输入固定 | **谨慎测试** | 首轮固定 280；后续仅 14 倍数如 `252,280,308` |
| `width` | 504 | `config.py:16`; `preprocessing.py:62` | 同上；width-lock resize | 主要决定缩放比例和横向点密度 | 同上；改 width 还改变 crop/pad 几何 | 增大显著变慢 | 增大显著上升 | B；strict compatible，训练输入固定 | **谨慎测试** | 首轮固定 504；后续 `448,504,560`，均为 14 倍数 |
| `pad_rgb` | ImageNet mean `(0.485,0.456,0.406)` | `preprocessing.py:14,63` | 竖向 padding | 宽画幅时填充无效区域 | 改变边界输入分布，可能污染几何/pose | 基本无 | 无 | B；训练 FOV 固定；公共 infer 未暴露 | **不建议修改** | 固定默认值 |
| resize/crop 策略 | bicubic+AA；中心裁剪/mean pad | `preprocessing.py:65-100` | 输入 FOV | 保持宽度、处理纵横比 | 与训练预处理强相关 | 算法差异很小 | 无 | B；无权重 shape 绑定但训练分布绑定 | **不建议修改** | 固定发布实现 |
| `amp_dtype` | bf16 | `config.py:14`; CLI | 输入、autocast、KV dtype | fp32/fp16/bf16 计算 | fp32可能更稳定但未必提升指标；fp16可能溢出；关键 point/Pose heads显式转 fp32 | bf16/fp16通常快于fp32 | fp16/bf16约低于fp32 | A/B；strict compatible，数值路径变化 | **建议测试** | `bf16, fp32`；显卡适合时补 `fp16` |
| `device` | cuda | `config.py:13` | 模型构造/执行 | 执行设备 | CPU/GPU数值可能略异 | CPU极慢 | GPU显存随设备选择 | A；无绑定 | **不建议修改** | 服务器固定 CUDA |
| `attention_backend` | auto | `config.py:19`; CLI | paged FlashInfer / SDPA | KV存储和 attention kernel | 理论可等价，仓库缺少 backend 数值 parity 证明 | paged通常更快；需实测 | paged预分配，SDPA维护 packed state；需实测峰值 | A；相同权重，真实 checkpoint 两后端 strict load 测试存在 | **强烈建议测试** | `paged, sdpa`；先做输出差异核对 |
| `ABOT_RECON_ROPE2D_BACKEND` | auto | `pos_embed.py:110` | 局部/交叉 attention 的 2D RoPE | CUDA cuRoPE 或 PyTorch fallback | 测试要求 CUDA实现与 reference exact | CUDA版主要改善速度 | 小 | A；环境变量，无权重绑定 | **建议测试** | `cuda` 对 `torch`，质量只需 parity 检查 |
| `local_window_frames` | 12 | `config.py:17,36`; `network.py:88`; attention/KV | global decoder causal KV | 当前+历史完整帧视觉 context | 小可能损失稳定性；大可能提供更多上下文但显著偏离训练 | 大会增加 attention 计算 | 近似线性增加完整帧 KV | B；无 tensor shape 绑定，但公开 config 强制 12，训练分布待验证 | **不建议修改** | 固定 `12`；研究性实验可旁路测 `8/16`，需明确非发布设置 |
| `max_frames` / `paged_max_total_frames` | 22000 | CLI/config；`model.py:97,102` | RoPE容量；可选 summary pool | 支持的最大时间/空间位置索引 | 容量足够时不改变预测；过小会越界/耗尽 | 容量足够时每帧计算不变 | 发布 `num_summary_tokens=0` 时主要是小型RoPE表，不随其预分配长程KV | A；真实 checkpoint 在 64 下 strict load 成功 | **建议测试** | 设为 `max(N, 5+H/14, 5+W/14)` 并留余量；如 1k/5k/实际N/22k |
| `output_local_points` | True | `config.py:20`; `api.py:57` | point decoder、返回值 | 返回相机坐标点图 | 是后续重投影、match/fusion最灵活的数据 | 关闭且不需world可跳过 point head | 关闭降低输出内存/存储 | D | **强烈建议测试** | 质量流程保持 True |
| `output_world_points` | False | `config.py:21`; `api.py:58` | point head + pose transform | 返回当前最终轨迹下的世界点 | 方便直接拼接，但受轨迹版本约束 | 增加变换和传输 | 显著增加CPU内存/存储 | D | **建议测试** | 评估/展示时开；长期资产优先保留 local points |
| `output_confidence` | True | `config.py:22`; `api.py:59` | conf decoder/head、返回值 | 保存每像素置信度 | 不改变Pose/点预测；对离线清噪很重要 | 关闭可跳过 conf decoder（threshold=0时） | 降低临时与输出内存 | D | **强烈建议测试** | 质量实验保持 True |
| `confidence_threshold` | 0.0 | `config.py:23`; `api.py:92-181` | sigmoid后 mask | 把低置信点置NaN | 大会降噪/飞点但降低覆盖率和可匹配点；小反之 | 已算 confidence 后几乎无影响 | 大会减少导出PLY，不减少原tensor大小 | A/D；纯后处理 | **强烈建议测试** | 从已有输出扫 `0,0.2,0.4,0.5,0.6,0.7,0.8,0.9`，再按分位数细化 |
| `point_z_log_max` | 10.0 | `model.py:94`; `depth_utils.py` | point head 后处理 | 只截断过大的 log-depth 再 exp | 降低可抑制极远溢出点，也会硬截断真实远景；只截上界 | 无实质影响 | 无 | B；strict compatible，但训练/发布固定 | **谨慎测试** | 通常固定10；极端远点诊断可测 `6,8,10` |
| `loop_closure` | True（CLI/API默认） | `config.py:24`; `api.py:99` | SALAD+reinfer+PGO | 用回访约束修正轨迹 | 有真实闭环时可减长程漂移；假阳性会扭曲全局点云 | 显著增加总时长 | SALAD与PGO占额外显存，worker与主模型并发 | E | **强烈建议测试** | `off/on`；先把 no-loop 当基线 |
| `auto_download` | False | `LoopClosureConfig:55` | loop资产解析 | 缺资产时是否下载 | 不影响预测 | 只影响首次准备时间 | 无 | D/E；高层固定False | **不建议修改** | 预先下载并校验资产 |
| loop checkpoint paths | 固定两个相对路径 | `config.py:25-26`; `LoopClosureConfig` | DINO/SALAD | 选择检索权重 | 错配会加载失败或 descriptor失真 | 仅加载开销差异 | 模型大小决定额外显存 | C/E；各自 checkpoint/架构绑定 | **不可修改** | 使用发布配套资产 |
| `loop_output_dir` | `outputs/loop` | config/CLI | loop JSON输出 | 保存候选、边、统计 | 不影响预测 | 少量I/O | 无GPU影响 | D/E | **建议测试** | 仅按实验目录修改 |
| `salad_backbone` | dinov2_vitb14 | `LoopClosureConfig:58`; `RetrievalConfig:28` | descriptor模型 | DINO backbone规格 | 改 backbone 会与 DINO/SALAD权重通道不匹配 | 模型大小决定速度 | 模型大小决定显存 | C/E；checkpoint绑定 | **不可修改** | 固定默认 |
| `salad_image_size` | (336,336) | `LoopClosureConfig:59` | descriptor预处理 | retrieval输入分辨率 | 改变 descriptor分布；且尺寸应兼容patch14 | 大会变慢 | 大会增加 | B/E；SALAD训练分布绑定 | **不建议修改** | 固定336×336 |
| `salad_batch_size` | 32 | `LoopClosureConfig:60`; worker默认32 | descriptor批处理 | 只改变descriptor吞吐 | eval下应基本不改变描述子 | 大批通常更快 | 大批增加峰值 | E；无权重绑定 | **建议测试** | `8,16,32,64` 按显存；公开API需改适配 |
| `descriptor_cache` | True | `LoopClosureConfig:61` | **未被读取** | 名义缓存开关 | 当前无作用 | 当前无作用 | 当前无作用 | E；死参数/待实现 | **不可修改** | 无 |
| `descriptor_queue_size` | 64 | `LoopClosureConfig:62` | **apply路径未读取** | 名义worker队列 | 当前通过低层 worker 的 `queue_size` 才可生效 | 只影响CPU/GPU流水阻塞 | 增大CPU tensor队列 | E；公开路径实际固定64 | **不建议修改** | 如改代码可测 `16,32,64` |
| `salad_score_threshold` | 0.85 | `LoopClosureConfig:63` → `RetrievalConfig.score_threshold` | 候选过滤 | 仅保留 score > threshold | 高：少假阳性但漏环；低：召回高但错误边风险大 | 低阈值增加re-inference直到上限 | 候选列表影响很小 | E；无绑定；公开API未暴露 | **强烈建议测试** | `0.80,0.85,0.90,0.93`，结合人工/几何审查 |
| `retrieval_top_k` | 5 | `LoopClosureConfig:64` → FAISS `search_k=top_k+1` | 每帧近邻检索 | 每个query检查的近邻数 | 大提高召回也增加重复/假候选 | 增加FAISS和后续候选成本 | 小幅增加搜索输出 | E；无绑定；公开API未暴露 | **建议测试** | `3,5,10,20` |
| `min_frame_separation` | 30 | `LoopClosureConfig:65` | retrieval及reinfer二次过滤 | 排除时间邻近帧 | 大可避免把普通相邻重叠当loop，但短回环会漏；其物理时间随输入stride变化 | 大减少候选 | 小 | E；单位是**已选输入帧** | **强烈建议测试** | `30,60,120`；按视频FPS/stride换算实际秒数 |
| `nms_radius` | 25 | `LoopClosureConfig:66` → candidate suppression | loop候选NMS | 抑制端点附近重复闭环 | 大提高多样性但可能删掉有效重复约束；小产生冗余 | 大减少reinfer/PGO | 小 | E；单位是已选输入帧 | **建议测试** | `10,25,50` |
| retrieval `keyframe_stride` | 1 | `LoopClosureConfig:67` | descriptor子采样、retrieval | 每N帧参与候选检索 | 大可能漏过短暂回访；小覆盖完整 | 当前worker仍算所有帧descriptor；只降低FAISS/候选/reinfer | descriptor峰值不变，候选规模下降 | E；公开API未暴露 | **建议测试** | `1,2,4,8`；质量优先从1开始 |
| `loop_chunk_size` | 10 | `LoopClosureConfig:68` | ABot camera-only reinference | 每个候选两端的局部上下文长度 | 大可能让相对Pose约束更稳，但改变局部序列起点/context且边仍只取端点Pose | 成本约随 `2*chunk_size*candidates` 增长 | 峰值主要由固定window决定；时长增加 | E/B；无shape绑定 | **强烈建议测试** | `6,10,12,20`；12与主window长度有解释性 |
| `max_candidates` | 1000 | `LoopClosureConfig:69` → NMS limit | retrieval结果上限 | 限制记录/排序候选 | 若高于reinfer上限，对最终PGO通常无影响 | 主要影响列表；最终成本受reinfer上限 | 很小 | E | **不建议修改** | 保持 `>= max_reinfer_candidates`；如需诊断100–1000 |
| `max_reinfer_candidates` | 50 | `LoopClosureConfig:70` | loop re-inference | 只验证score最高的N个候选 | 大增加约束覆盖，也会把较弱候选直接变成边；当前没有几何inlier验证 | **主要线性成本旋钮** | 峰值变化小，总GPU时间大增 | E；公开API未暴露 | **强烈建议测试** | `10,25,50,100`，先看候选精度 |
| `default_inliers` | 128 | `LoopClosureConfig:71` | loop edge权重 | 给所有reinfer边写固定inliers | 不是真实inlier计数；会通过 `sqrt(inliers/rigid_min_inliers)` 放大loop权重 | 无 | 无 | E；启发式 | **不建议修改** | 固定128，优先直接调loop_weight |
| `pose_graph_loop_weight` | 0.01 | `LoopClosureConfig:72`; `PoseGraphConfig:18` | PGO edge weights | loop边相对odom边权重，再乘sqrt(inliers/24) | 大：更强纠漂但假环破坏更重；小：更保守 | 几乎不改变规模 | 无 | E；纯优化参数 | **强烈建议测试** | 对数尺度 `0.0025,0.005,0.01,0.02,0.05` |
| `pose_graph_node_mode` | sparse_keyframes | `LoopClosureConfig:73` | graph构建 | sparse模式抽节点；其他字符串实际落入全帧图 | 全帧更细但未必更准；sparse correction平滑插值 | 全帧PGO显著慢 | 全帧更高 | E；代码未严格校验枚举 | **建议测试** | `sparse_keyframes`；小片段可与任一非该字符串触发的full对照，建议先补显式枚举 |
| `pose_graph_keyframe_stride` | 50 | `LoopClosureConfig:74` | sparse graph节点 | 每N帧保留规则节点，并强制加入loop端点 | 大使校正更平滑/粗糙，可能欠拟合局部漂移；小更灵活 | 小值增加PGO成本 | 小值增加PGO状态 | E；不影响retrieval keyframes | **强烈建议测试** | `20,50,100,200` |
| `pose_graph_trans_weight` | 1.0 | `LoopClosureConfig:75` | PGO residual | 缩放平移残差分量 | 相对rot_weight决定优化侧重；绝对同时缩放会改变阻尼相对量级 | 规模不变 | 无 | E | **建议测试** | 固定trans=1，测 rot/trans 比 `0.5,1,2` |
| `pose_graph_rot_weight` | 1.0 | `LoopClosureConfig:76` | PGO residual | 缩放旋转残差分量 | 大更重视方向一致，可能改善远处点云拼接，也可能牺牲位置 | 规模不变 | 无 | E | **强烈建议测试** | `0.5,1,2,4`（相对trans） |
| `pose_graph_max_iterations` | 30 | `LoopClosureConfig:77` | LM外迭代 | PGO最大外循环 | 太小不收敛；足够后继续增加通常无收益 | 增大上限最坏线性变慢，有早停 | 基本不变 | E | **建议测试** | `10,20,30,50`，依据cost/convergence日志 |
| `pose_graph_lambda_init` | 1e-6 | `LoopClosureConfig:78` | PGO damping | 初始阻尼 | 过小在坏初值下可能不稳；过大更新保守 | 可能影响接受/拒绝次数 | 无 | E | **谨慎测试** | `1e-7,1e-6,1e-5,1e-4` |
| `gpu_pgo_pcg_max_iterations` | 256 | `LoopClosureConfig:79` | 每次线性求解 | PCG迭代上限 | 太小导致线性解不准 | 上限增大最坏更慢 | 基本不变 | E | **建议测试** | `64,128,256`，先看统计中的收敛标志 |
| `gpu_pgo_pcg_tolerance` | 1e-5 | `LoopClosureConfig:80` | PCG停止条件 | 相对残差目标 | 更小解更精确，但未必改善最终轨迹 | 更小通常更慢 | 无 | E | **谨慎测试** | `1e-4,1e-5,1e-6` |
| `gpu_pgo_pcg_check_interval` | 8 | `LoopClosureConfig:81` | PCG检查 | 每N步同步检查残差 | 不直接改目标；大会延迟停止 | 大减少同步但可能多迭代 | 无 | E | **不建议修改** | 固定8 |
| `gpu_pgo_coarse_group_size` | 64 | `LoopClosureConfig:82` | 两级block-Jacobi预条件 | coarse grouping | 主要影响求解效率/数值 | 数据规模相关 | coarse矩阵随group数变化 | E | **谨慎测试** | `32,64,128`，仅大图性能实验 |
| `gpu_pgo_solve_dtype` | float64 | `LoopClosureConfig:83` | PGO线性代数 | float64/其他按float32处理 | float64更稳；float32可能损伤大图收敛 | float32可能更快 | float32更低 | E | **建议测试** | 质量优先固定float64；只为速度测float32 |
| `faiss_use_gpu` | True | `RetrievalConfig:36` | FAISS | descriptor索引设备 | IndexFlatIP结果应近似一致，边界分数可能受数值影响 | GPU通常快 | 占额外GPU显存 | E | **建议测试** | GPU/CPU性能对照 |
| `faiss_gpu_id` | -1 | `RetrievalConfig:37` | FAISS | -1跟随inference device index，否则指定卡 | 无质量作用 | 资源调度 | 转移显存占用 | E/D | **建议测试** | 多GPU时把SALAD/FAISS放独立卡需改worker设计 |
| `faiss_query_batch_size` | 512 | `RetrievalConfig:38` | FAISS search loop | 查询分批 | 不应改变结果 | 吞吐旋钮 | 批越大临时内存越高 | E | **建议测试** | `128,256,512,1024` |
| `faiss_require_gpu` | False | `RetrievalConfig:39` | fallback策略 | GPU失败时是否报错 | 无质量作用 | 防止静默CPU降速 | 无 | E/D | **不建议修改** | benchmark可设True保证口径 |
| loop/retrieval `verbose` | True | 两个loop config | 日志 | 打印阶段统计 | 不影响结果 | 少量终端I/O | 无 | D/E | **不建议修改** | 调试开，批量实验可关 |
| benchmark `warmup_frames` | 2 | `scripts/benchmark_video_reconstruction.py:30` | benchmark预热 | 正式计时前运行短序列 | 不影响正式输出 | 只影响测量稳定性和总墙钟时间 | 预热后清cache并重置峰值 | D | **建议测试** | 固定2–10并记录；不作为模型超参 |
| `output point_stride` | 4 | `export_reconstruction_ply.py:33` | PLY导出 | 每N像素采一点 | 大降低密度、可减冗余与文件体积；规则网格可能影响registration细节 | 导出/后处理更快 | CPU/文件显著下降 | D；完全离线 | **强烈建议测试** | `1,2,4,8`；match/fusion建议先保留1/2 |
| `output frame_stride` | 1 | PLY导出:34 | PLY导出 | 每N个dense frame导出 | 大降低跨视角覆盖与重复点 | 更快 | CPU/文件下降 | D；完全离线 | **建议测试** | `1,2,4`；与dense_stride不要重复过度抽样 |
| `output max_points` | 5,000,000 | PLY导出:35 | PLY导出 | linspace确定性截断总点数 | 过低损伤覆盖；不是几何感知采样 | 降低写入/下游成本 | CPU/文件下降 | D；完全离线 | **强烈建议测试** | `1M,5M,10M,不限制(<=0)`；正式fusion优先体素采样替代 |
| `points_frame` | auto | PLY导出:25 | 坐标变换 | 指明输入点图local/world | 设置错误会重复或漏做Pose变换 | 无 | 无 | D | **不可修改** | 必须与文件真实坐标系一致 |
| `pose_stride/frustum_scale` | 1 / 0.15 | PLY导出:36-37 | 相机视锥可视化 | 轨迹相机符号密度/大小 | 不影响点云顶点和模型预测 | 只影响少量导出 | 无 | D | **不建议修改** | 按可视化需要 |
| `bev_size/bev_plane` | 1600 / auto | PLY导出:39-45 | BEV PNG | 轨迹图尺寸与投影平面 | 不影响模型/PLY点云 | 只影响绘图 | 无GPU影响 | D | **不建议修改** | 按可视化需要 |

## 5. 固定模型结构与训练参数

以下参数在 `model.py:ReleasedABotReconModel` 中硬编码，不能当普通推理超参数。表中“不可修改”表示不应在已发布 checkpoint 上直接改变；其中部分值在技术上可能通过非 strict 加载凑合运行，但那会得到随机初始化、缺层或语义错配的模型。

| 参数组 | 发布值 | 绑定证据与实际影响 | 分类/建议 |
|---|---|---|---|
| `decoder_size`, `decoder_depth_override` | large / None，即dim1024、16 heads、36 blocks | 改 decoder宽度/深度会大面积改变或增删 state_dict；checkpoint含36层decoder权重 | C / **不可修改** |
| `gate_layers` | `0..35` 全层 | 每层注册 `gate_proj.weight [1024,1024]`；checkpoint实际含36个该key | C / **不可修改** |
| `pos_type` | rope100 | 改变所有2D rotary的频率语义；虽RoPE本身无持久权重，也会严重偏离训练 | B/C语义绑定 / **不建议修改** |
| `global_pos_encoding` | rope3d | checkpoint metadata明确 `position_encoding=rope3d`；改为2D/none不一定制造shape mismatch，但改变全局attention语义 | B/C语义绑定 / **不可修改** |
| `rope3d.theta` | 10000 | 不产生持久权重，但改变时间/空间旋转频率 | B/C训练语义绑定 / **不可修改** |
| `rope3d.fhw_dim` | `[20,22,22]` | 三者总和必须等于head_dim64；改分配改变训练位置编码语义，和不合法值会构造失败 | C语义/运行绑定 / **不可修改** |
| `rope_fwd_cache_max` | 32（RoPE3D类默认） | 只控制按shape/device缓存的RoPE输出，理论是性能参数；发布构造未暴露 | A/D，但收益小 / **不建议修改** |
| `camera_pose_mode` / head type | relative_adjacent / token_pair | 改 absolute 会换成完全不同CameraHead并产生大量key不匹配 | C / **不可修改** |
| `rotation_format` / `translation_param` | quat / vector | 构造器明确拒绝其他值；输出头和Pose解释与checkpoint绑定 | C / **不可修改** |
| `hidden_dim`, `pair_hidden_dim` | 512 / 512 | 改变frame descriptor、pair MLP、输出head shape | C / **不可修改** |
| `num_pose_tokens` | 5 | 决定从5个register token聚合Pose；checkpoint `register_token [1,1,5,1024]`，模型自身也固定5 | C / **不可修改** |
| `init_std` | 1e-4 | 只影响新模型初始化，加载完整checkpoint后被覆盖 | F / **不建议修改** |
| `rot_correction_mode` | temporal_rotation_refinement | 构造器拒绝其他值；去掉模块会产生checkpoint key差异 | C / **不可修改** |
| `rot_correction_kernel` | 10 | 真实checkpoint shape直接绑定三个tensor；已验证kernel8 strict load失败 | C / **不可修改** |
| `rot_correction_hidden_dim` | 512（None回退hidden） | 改变refiner几乎所有线性、attention和卷积tensor shape | C / **不可修改** |
| `rot_correction_use_age_embed` | True | checkpoint实际含 `age_embed.weight`；False时已验证strict load unexpected key | C / **不可修改** |
| refiner `num_heads` | 8（调用点硬编码） | 在hidden_dim不变时MultiheadAttention参数shape可能不变，但head拆分与训练语义改变 | C语义绑定 / **不可修改** |
| `rot_correction_max_deg` | 2.0 | 无state tensor；已验证改为1.0 strict load成功，但直接重标定逐帧旋转校正 | B / **谨慎测试**，范围 `1,1.5,2,2.5,3°` |
| `use_global_points` | False | True会新增global decoder/head，checkpoint无这些key | C / **不可修改** |
| `enable_confidence` | 根据checkpoint前缀自动检测 | 发布checkpoint含conf decoder/head；强制False会产生unexpected keys，True用于无conf checkpoint则missing keys | C / **不可修改** |
| `train_conf`, `confidence_only_train`, `init_conf_decoder_from_point` | False / False / False | 构造/冻结/初始化训练控制；推理完整checkpoint无调节意义 | F / **不建议修改** |
| `freeze_encoder`, `freeze_prediction_heads` | False / False | 只控制requires_grad，不影响inference_mode结果 | F / **不建议修改** |
| `num_dec_blk_not_to_checkpoint` | 4 | 只影响训练activation checkpointing条件 | F / **不建议修改** |
| `load_vggt`, `ckpt`（Pi3内部） | False / None | 初始化/训练warm-start路径；发布外层另行 strict load完整ABot checkpoint | F/C / **不可修改** |
| `causal_global_attn` | True | 流式paged路径天然只看过去；SDPA路径也按causal streaming工作。关闭会偏离模型定义，公开infer不暴露 | B/C语义绑定 / **不可修改** |
| `infer_mode` | stream | iterable API只允许stream；full需要完整tensor并走不同内存/调用路径 | B / **不建议修改** |
| `memory_mode` | streaming | window/streaming决定被驱逐帧是压缩summary还是丢弃；发布时summary token=0，两者仍有内部差异和训练语义 | B/C / **不可修改** |
| `num_reference_frames` | 0 | 改变长期可见帧集合，无权重shape但改变context分布 | B/C / **不可修改** |
| `num_summary_tokens` | 0 | 改变旧帧压缩KV可见性；无新增学习参数但发布模型未使用长期summary | B/C / **不可修改** |
| `paged_max_summary_frames` | 0 | summary token=0时无效；只有启用未训练summary机制才控制FIFO cap | B/死路径 / **不可修改** |
| `paged_force_fp32` | False | debug gather+SDPA路径，不是正常FlashInfer backend；会增加内存且路径不同 | A但仅验证用途 / **不建议修改** |
| `use_packaged_flash_attn` | False | 构造局部attention实现选择；发布外层还会禁用不可用模块 | A/B内部实现 / **不建议修改** |
| attention dropout/proj dropout/drop path | 0 | eval下dropout为0；改变主要是训练正则 | F / **不建议修改** |
| `pose_graph_model/update_mode` | se3 / all | `PoseGraphConfig`存在，但`LoopClosureConfig`未暴露，当前调用保持SE(3)全量更新；sim3会增加scale自由度 | E/C语义 / **不建议修改** |
| `pose_graph_scale_weight` | 1.0 | 仅sim3模型缩放scale residual；当前se3路径完全不生效 | E/死参数 / **不可修改** |
| `rigid_min_inliers` | 24 | 进入loop edge权重分母；当前loop边的inliers又被固定为128，二者共同形成启发式放大 | E；未由高层传入 / **不建议修改** |
| `gpu_pgo_outer_relative_tolerance` | 0.0 | 大于0才启用相对cost早停；当前调用未从LoopClosureConfig传入 | E；solver参数 / **谨慎测试**，但应先暴露并记录 |
| `pose_graph_solver_verbose` | False | 只控制PGO日志，当前LoopClosureConfig.verbose不传给它 | D/E / **不建议修改** |
| SALAD `num_clusters/cluster_dim/token_dim` | 64 / 128 / 256 | 决定aggregator层shape和最终descriptor；SALAD checkpoint绑定 | C/E / **不可修改** |
| Sinkhorn `iterations` | 3 | descriptor assignment固定迭代数；无公开config，改变会改变SALAD descriptor分布 | B/E训练语义 / **不建议修改** |

## 6. “是否真的可调”的四档判断

### 安全可调

- `stride`、`start/end`：代码直接切片，不触碰模型/checkpoint；必须重新推理。
- `dense_stride/dense_output_indices`：不改变Pose路径；必须重新推理才能补回此前未计算的dense帧。
- `confidence_threshold`：已有原始点图和confidence时完全离线可调。
- dense输出开关：strict load与forward均安全，影响head是否执行和保存内容。
- `attention_backend`：相同模型shape，两个backend均支持strict load；正式质量实验前需验证输出数值差异。
- `max_frames`：不改变权重shape；只要大于所有实际时间/空间位置索引即可。
- PLY `point_stride/frame_stride/max_points`：纯离线输出参数。
- loop内部 retrieval/PGO 数值参数：只要通过低层API正确传入，不影响主checkpoint；它们应在loop专项阶段测试。

### 谨慎可调

- `amp_dtype`：forward能运行、shape不变，但数值误差和硬件kernel不同。
- `height/width`：strict load可以，forward要求14整除；改变DINO位置插值、token数、裁剪/pad和训练FOV分布。
- `rot_correction_max_deg`：strict load实测成功，权重shape不变；却直接缩放训练输出并累计到长程Pose。
- `point_z_log_max`：只在point head后处理，但会截断真实深度。
- `loop_chunk_size`：不改权重，却改变loop re-inference的上下文和起点。
- PGO阻尼、容差、精度：不会破坏主模型，但可能导致优化不收敛或收敛到不同结果。

### 不建议修改

- `local_window_frames`：底层可改、不会造成state_dict size mismatch，但公开release主动拒绝非12；训练分布证据不足。
- 发布预处理策略、`pad_rgb`、RoPE频率/分配、causal/memory/reference/summary设置：多数不一定触发strict-load错误，却改变训练时定义的输入/context语义。
- `max_candidates` 高于 `max_reinfer_candidates` 时不会增加最终loop edge，只增加候选记录，实验收益低。
- refiner初始化、训练冻结、activation checkpointing参数：推理无意义。

### 不可修改

- `rot_correction_kernel`、`rot_correction_hidden_dim`、`rot_correction_use_age_embed`。
- decoder宽深、gate layers、camera head模式、pose token数量、confidence/global point分支结构。
- DINO/SALAD backbone与不匹配的loop checkpoints。

## 7. 面向高质量跨视频 match/fusion 的优先级

### 第一优先级

1. **`confidence_threshold`**：直接控制飞点与覆盖率，且可以对同一份原始输出零成本反复扫描。应同时记录保留点比例、几何指标和跨视频registration内点数，不能只看视觉整洁。
2. **`stride`**：决定模型实际看到的相邻运动和重叠，是Pose稳定性和点云覆盖最上游的安全变量。服务器质量优先应以1为主，并用2/3量化速度换质量。
3. **`dense_stride`**：独立于Pose采样，能在不损失Pose context的情况下控制点图密度、dense head时间和存储。对fusion建议先比较1和2，再看4。
4. **`amp_dtype`**：用bf16作为发布基线，对一段困难序列补fp32，检查长程Pose累计差异和点深尾部；如果差异不可测，则保持bf16节省时间/显存。
5. **`attention_backend`**：paged与SDPA先做小样本输出差异和性能对照；确认质量容差后选服务器上更快/更省显存的一种。
6. **离线点云抽样参数 `point_stride/max_points`**：它们直接决定registration/fusion可用点数，却无需重新推理。规则抽样的基线之外，后续最好加入体素采样和每体素confidence保留策略。

这里不把 `height/width` 放入首轮：发布模型明确围绕280×504训练/评测，改变它的风险高于普通“分辨率越高质量越好”的直觉，而且现有CLI颜色保存还会shape错配。

### 第二优先级

1. **`rot_correction_max_deg`**：可能影响长程旋转漂移，但它是训练输出标定，不宜早于安全参数。用1–3°小范围，并评估每帧RPE-R及长序列ATE/闭环误差。
2. **`point_z_log_max`**：仅在发现极端远点/overflow时测试；正常序列默认10很少触发。
3. **`height/width`**：只在默认设置的主要实验完成后进行，保持14整除并同步修复colors预处理。必须同时看Pose、点图、FPS与VRAM。
4. **`max_frames`**：主要用于把容量和少量RoPE缓存调到实际任务长度，不期待改善质量。
5. **`output_local/world/confidence`**：规划数据产品时调整。跨视频fusion推荐保存local points、confidence和最终/原始两套Pose；world points可随时重建。

### 第三优先级：开启 Loop Closure 后

按依赖顺序测试：

1. `loop_closure off/on`，先确认视频确有回访且默认loop没有假阳性。
2. `salad_score_threshold` + `min_frame_separation`，先控制候选精度与时间语义。
3. `retrieval_top_k` + `nms_radius` + retrieval `keyframe_stride`，控制召回、多样性和候选规模。
4. `max_reinfer_candidates` + `loop_chunk_size`，这是回环额外ABot推理成本的主要组合。
5. `pose_graph_loop_weight` + `pose_graph_rot_weight/trans_weight`，控制约束真正如何改变轨迹。
6. `pose_graph_keyframe_stride`，控制校正自由度与PGO规模。
7. solver iterations/tolerance/dtype，仅在日志表明未收敛或PGO成为瓶颈时测试。

### 不建议动

所有 C 类结构参数、非12 local window、RoPE语义、训练冻结/初始化参数，以及没有调用点的 `descriptor_cache`。这些变量即使名字看似“可调”，也不应进入普通推理超参表。

## 8. 推荐单因素实验顺序

建议先锁定一组包含慢速旋转、快速运动、低纹理、远景和真实回访的代表视频，所有实验保留 `camera_poses_noloop`、local points、confidence及metadata。

1. **Baseline**：`stride=1, dense_stride=1, 280×504, bf16, attention_backend=当前auto解析结果, local_window=12, confidence_threshold=0, loop=off`。先输出未经筛选的原始数据。
2. **confidence离线扫描**：从同一 baseline 的 `local_points.pt + confidence.pt` 扫 `0–0.9`，无需推理。选取兼顾F1/Chamfer、保留率和registration内点的阈值。
3. **PLY/融合离线采样**：`point_stride=1/2/4/8`、`max_points`，无需推理。不要把导出密度变化误记为模型点图质量变化。
4. **stride**：重新推理 `1/2/3`。每个stride都按原视频帧率换算RPE时间间隔；loop的min separation也应随后换算。
5. **dense_stride**：重新推理 `1/2/4/8`。Pose应保持一致；主要评价dense覆盖、整体FPS、输出体积和fusion效果。
6. **attention backend**：重新加载模型并重新推理 paged/SDPA；先用短片段检查Pose、point/conf差异，再量FPS/VRAM。
7. **amp dtype**：重新加载模型并重新推理 bf16/fp32（必要时fp16）。比较长程累计Pose而不只看单帧。
8. **rotation bound**：需改构造参数、重新加载并重新推理 `max_deg=1/1.5/2/2.5/3`。这是研究性B类实验，2°始终作为发布基线。
9. **分辨率**：需改API config和颜色保存代码、重新加载/推理。只在前述结果稳定后做小网格。
10. **loop off/on**：基础模型已有no-loop Pose和local points时，理论上loop可作为后处理单独运行；当前高层API会在同一次调用中启动descriptor worker并执行loop。为避免重复基础推理，正式benchmark前建议增加“从已保存Pose/descriptor运行 `apply_loop_closure`”的实验入口。
11. **loop retrieval → reinference → PGO**：依照上一节第三优先级逐层单因素，不要同时改阈值、候选上限和loop weight，否则无法判断假阳性来自何处。

按重跑需求归纳：

| 操作 | 参数 |
|---|---|
| 无需模型重跑，已有原始输出即可 | `confidence_threshold`、world变换所用Pose选择、PLY `point_stride/frame_stride/max_points` |
| 必须重新推理，但可复用同一已加载模型 | `stride`、`dense_stride`、输出head组合；同尺寸/同backend下的不同片段 |
| 应重新构造/加载模型 | `amp_dtype`、`attention_backend`、`max_frames`、`height/width`、研究性的`rot_correction_max_deg` |
| 可对已有base Pose做独立后处理，但当前需低层入口 | 所有retrieval/loop/PGO参数 |
| 需要重新训练或至少专门迁移权重并微调 | `rot_correction_kernel/hidden_dim/use_age_embed`、decoder/camera head/gate/pose tokens、confidence/global point结构 |

## 9. 对最终七个问题的回答

### 1. 真正适合作为推理调参的参数

首选是 `stride`、`dense_stride/dense_output_indices`、`confidence_threshold`、`amp_dtype`、`attention_backend`、合理的`max_frames`和各输出开关。Loop开启后，retrieval threshold/top-k/separation/NMS、reinfer上限/chunk、PGO权重和keyframe stride也是真正的后处理超参，但当前高层API没有暴露它们。

### 2. 能改但不应直接当普通推理超参的参数

`local_window_frames`、`height/width`、`rot_correction_max_deg`、`point_z_log_max`、RoPE theta/fhw分配、memory/reference/summary模式。前四者至少在技术上可以保持strict load，但会改变训练时固定的context、FOV或输出标定；后几者直接改变模型语义。

### 3. 与训练/checkpoint 强绑定的参数

最明确的是 `rot_correction_kernel=10`、age embedding开关、refiner hidden dim、decoder size/depth、36层gate、相邻Pose head模式、5个pose tokens、confidence分支存在性、DINO/SALAD backbone与权重。`local_window_frames=12` 是训练/context绑定而非tensor shape绑定，这一区别必须保留。

### 4. 最可能改善最终点云质量的参数

在不重训条件下，最可靠的是 confidence筛点、保持低输入stride、保持较密dense输出，以及有真实回访时正确配置loop closure。分辨率增大并不保证改善，因为模型训练FOV固定；rotation bound可能改善或恶化长程漂移，必须谨慎验证。

### 5. 主要改善速度的参数

`stride` 的提速最大但会改变Pose问题；`dense_stride` 在保留Pose帧率时跳过point/conf heads；关闭不需要的dense输出也会提速。`attention_backend=paged`、cuRoPE、bf16/fp16主要优化kernel与内存。`max_frames`不是FPS旋钮。

### 6. 对跨视频点云 match/fusion 最重要的参数与数据

最重要的是：低`stride`保证视角重叠与Pose连续性，合适的confidence筛选去飞点，足够小的`dense_stride`保留表面覆盖，稳定的全局Pose/loop结果，以及保留**local points + confidence + 原始与loop Pose**。离线 `point_stride/max_points` 决定registration输入密度。对不同视频，应使用一致的预处理、阈值和坐标约定；当前loop只做单序列轨迹优化，不能自动解决跨视频坐标对齐。

### 7. 下一步最值得测试的 5–10 个参数

按当前“服务器优先质量、稳定和fusion”的目标排序：

1. `confidence_threshold`: `0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`，先离线扫。
2. `stride`: `1,2,3`，以1为主。
3. `dense_stride`: `1,2,4,8`，质量候选优先1/2。
4. PLY `point_stride`: `1,2,4,8`，离线评估registration/fusion。
5. `attention_backend`: `paged,sdpa`，先做数值差异核对。
6. `amp_dtype`: `bf16,fp32`，显卡/数据需要时加fp16。
7. `loop_closure`: `off,on`，只在有真实回访的视频上评估。
8. `salad_score_threshold`: `0.80,0.85,0.90,0.93`。
9. `max_reinfer_candidates × loop_chunk_size`: 先单因素 `10/25/50` 与 `6/10/12/20`。
10. `pose_graph_loop_weight` 和 `pose_graph_keyframe_stride`: 分别 `0.0025–0.05` 对数尺度、`20/50/100/200`。

`rot_correction_max_deg=1–3°` 可作为第二轮第一个研究性参数，但不应挤占上述安全参数的首轮预算。

## 10. Checkpoint 与最小验证记录

使用只读公开 checkpoint：

```text
/data/user24302666/abot_navigation_comparison/20260901_224942_inventory/caches/abot_model/abot_recon.safetensors
文件大小：4,003,866,352 bytes
checkpoint tensor keys：1,274
metadata：model=ABot-Recon, position_encoding=rope3d, format=pt
```

验证结果：

- 完整模型以 `device=cpu, attention_backend=sdpa, amp_dtype=fp32, max_frames=64` 构造并 `strict=True` 加载成功；参数量 `1,000,934,170`，运行时 `rope3d.max_seq_len=64`、`local_window_frames=12`。这验证了 `max_frames` 不参与持久权重shape。
- checkpoint真实tensor：`camera_head.rot_correction.age_embed.weight [10,512]`、`conv.weight [512,1,10]`、`gate.weight [512,1,10]`。
- 用真实rotation-refiner子state_dict构造 `kernel_size=8` 后 strict load，三处均发生size mismatch。
- 构造 `use_age_embed=False` 后 strict load，发生 unexpected `age_embed.weight`。
- 构造 `max_rot_deg=1.0, kernel_size=10` 后 strict load，所有keys匹配成功。
- checkpoint实际包含 `register_token [1,1,5,1024]`、`point_head.proj.weight [588,1024]`、`conf_head.proj.weight [196,1024]` 和36个 `decoder.*.attn.gate_proj.weight [1024,1024]`，与发布构造完全对应。
- 当前执行环境 `nvidia-smi` 无法连接驱动，因此本阶段未做GPU forward、paged-vs-SDPA数值验证或完整benchmark；这符合本阶段只做源码分析和最小load验证的范围。涉及实际FPS/VRAM和非默认分辨率forward的结论仍需下一阶段验证。

## 11. 待验证事项与实现限制

- 训练代码/recipe未发布，无法证明训练时 `local_window_sample_values` 是否只含12；公开release和eval均强制12，因此本报告按“训练/context绑定，非tensor shape绑定”处理。
- 当前没有 attention backend 端到端数值parity测试；只有解析逻辑、真实checkpoint strict load和cuRoPE reference exact测试。
- 高层API无法传入 `LoopClosureConfig`，正式loop参数实验前应先增加可审计的配置入口，否则改dataclass默认值容易造成实验记录不透明。
- `descriptor_cache` 没有调用点；`descriptor_queue_size`、多数FAISS参数也没有从高层config传入。
- loop候选通过SALAD阈值后就被ABot re-inference结果直接作为edge，`default_inliers`是常数，并没有额外几何inlier验证。这使 `salad_score_threshold`、`max_reinfer_candidates` 与 `pose_graph_loop_weight` 对假阳性尤其敏感。
- 默认 `loop_closure=True`，但缺少loop可选依赖或资产会直接失败。做基础模型参数实验时应显式设为False，避免把SALAD/PGO开销与模型FPS混在一起。
- 非默认 `height/width` 时，CLI/demo保存colors仍固定280×504；需先修正后再进行可靠的点云导出与fusion比较。
