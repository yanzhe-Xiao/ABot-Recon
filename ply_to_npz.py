import open3d as o3d
import numpy as np
import sys

input_ply = sys.argv[1]
output_npz = sys.argv[2]

pcd = o3d.io.read_point_cloud(input_ply)

points = np.asarray(pcd.points).astype(np.float32)

colors = np.asarray(pcd.colors)

# Open3D 读取颜色通常是 0~1 浮点数，转成 0~255 uint8
colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

# PLY 里如果没有 confidence，就先全部设为 1
confidence = np.ones(len(points), dtype=np.float32)

np.savez_compressed(
    output_npz,
    points=points,
    colors=colors,
    confidence=confidence,
)

print("points:", points.shape)
print("colors:", colors.shape)
print("confidence:", confidence.shape)
print("saved:", output_npz)
