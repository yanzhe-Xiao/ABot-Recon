我需要使用R3PM-Net将两个局部子点云图（源点云source和目标点云target）进行拼接（点云配准）。请完成以下任务：

1.  **环境准备**：从官方GitHub仓库（https://github.com/YasiiKB/R3PM-Net）克隆代码[reference:0][reference:1]，并按照README创建conda环境。

2.  **数据加载**：我的两个点云文件分别为 `/home/data/xyz/ABot-Recon/outputs/mine_VID20260903181931_loop/reconstruction.ply` 和 `/home/data/xyz/ABot-Recon/outputs/mine_VID20260903182041_loop/reconstruction.ply`。将其转换为R3PM-Net所需的输入格式

3.  **模型推理**：加载R3PM-Net的预训练模型，将源点云和目标点云输入网络，获取输出的刚体变换（旋转矩阵 `R` 和平移向量 `t`）[reference:4][reference:5]。

4.  **点云拼接**：将刚体变换应用到源点云上，使其与目标点云对齐，完成拼接。

5. **结果保存**：将拼接后的点云保存为 `merged.ply` 文件。并且符合本项目的前端展示，可以作为选项展示出来。