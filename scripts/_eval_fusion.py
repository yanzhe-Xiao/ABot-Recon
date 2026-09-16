import json
from pathlib import Path
import numpy as np
import open3d as o3d
r=Path('/home/data/xyz/ABot-Recon/outputs')
for name in ['测试_merged_general_consensus','测试_merged_trajectory_prior']:
 m=json.load(open(r/name/'fused_2_streams_transform.json'))
 seqs=m['sequences']; anchor=m['anchor_sequence']; other=[x for x in seqs if x!=anchor][0]
 a=np.asarray(o3d.io.read_point_cloud(str(r/anchor/'reconstruction.ply')).voxel_down_sample(.03).points)
 b=np.asarray(o3d.io.read_point_cloud(str(r/other/'reconstruction.ply')).voxel_down_sample(.03).points)
 T=np.array(seqs[other]['transform_matrix']); s=seqs[other]['scale_to_anchor']; b=(s*b)@T[:3,:3].T+T[:3,3]
 pa=o3d.geometry.PointCloud(); pb=o3d.geometry.PointCloud(); pa.points=o3d.utility.Vector3dVector(a); pb.points=o3d.utility.Vector3dVector(b)
 print(name,'anchor',anchor,'other',other,'points',len(a),len(b))
 for d in [.03,.05,.08,.15,.3]:
  e=o3d.pipelines.registration.evaluate_registration(pb,pa,d)
  print(' ',d,round(e.fitness,4),round(e.inlier_rmse,4))
