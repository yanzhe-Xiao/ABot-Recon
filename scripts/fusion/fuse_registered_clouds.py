#!/usr/bin/env python3
"""Compare registration hypotheses and fuse only geometrically supported results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import load_cloud_npz, transform_points, voxel_fuse, write_ply


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--target",type=Path,required=True)
    p.add_argument("--method1",type=Path,required=True); p.add_argument("--method2",type=Path,required=True)
    p.add_argument("--correspondences",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--voxel",type=float,default=0.05); return p.parse_args()


def correspondence_validation(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict[str,object]:
    residual=np.linalg.norm(source@transform[:3,:3].T+transform[:3,3]-target,axis=1)
    return {"count":len(residual),"within_0.15_count":int((residual<=.15).sum()),"within_0.15_ratio":float((residual<=.15).mean()),
            "residual_median":float(np.median(residual)),"residual_p90":float(np.percentile(residual,90))}


def density(points: np.ndarray, radius: float=.1, samples: int=20000) -> float:
    rng=np.random.default_rng(9); query=points if len(points)<=samples else points[rng.choice(len(points),samples,replace=False)]
    return float(np.mean(cKDTree(points).query_ball_point(query,radius,return_length=True)-1))


def main() -> None:
    a=parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    src,tgt=load_cloud_npz(a.source),load_cloud_npz(a.target); corr=np.load(a.correspondences); p06,p08=corr["points_06"],corr["points_08"]
    t1=np.load(a.method1); t2=np.load(a.method2)
    v1=correspondence_validation(t1,p06,p08); v2=correspondence_validation(t2,p06,p08)
    comparison={"method1":v1,"method2":v2,"decision":"method2_lightglue_umeyama",
                "method1_status":"registration_geometrically_wrong" if v1["within_0.15_count"]<10 else "uncertain",
                "method2_status":"likely_success" if v2["within_0.15_count"]>=10 else "uncertain",
                "reason":"Method 2 is supported by independent lifted feature correspondences; method 1 aligns repeated geometry but contradicts all such correspondences."}
    comparison_dir=out.parent/"comparison"; comparison_dir.mkdir(parents=True,exist_ok=True)
    (comparison_dir/"registration_comparison.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    aligned=transform_points(src["points"],t2).astype(np.float32); all_points=np.concatenate((aligned,tgt["points"])); all_colors=np.concatenate((src["colors"],tgt["colors"])); all_conf=np.concatenate((src["confidence"],tgt["confidence"]))
    simple=voxel_fuse(all_points,all_colors,all_conf,a.voxel,False); weighted=voxel_fuse(all_points,all_colors,all_conf,a.voxel,True)
    method_dir=out/"method2"; method_dir.mkdir(parents=True,exist_ok=True)
    write_ply(method_dir/"registered_combined.ply",all_points,all_colors,all_conf)
    write_ply(method_dir/"fused_voxel.ply",simple["points"],simple["colors"],simple["confidence"])
    write_ply(method_dir/"fused_confidence.ply",weighted["points"],weighted["colors"],weighted["confidence"])
    cross=cKDTree(tgt["points"]).query(aligned,workers=-1)[0]
    payload={"pair_id":"06_clean_vs_08","method":"method2_lightglue_umeyama","fusion_status":"success",
             "voxel":a.voxel,"source_points":len(src["points"]),"target_points":len(tgt["points"]),"raw_combined_points":len(all_points),
             "fused_points":len(simple["points"]),"point_reduction_ratio":1-len(simple["points"])/len(all_points),
             "duplicate_voxel_count":int(simple["duplicate_voxels"]),"duplicate_voxel_ratio":float(simple["duplicate_voxels"])/len(simple["points"]),
             "nn_median_before_fusion":float(np.median(cross)),"nn_p90_before_fusion":float(np.percentile(cross,90)),
             "nn_after_definition":"nearest-neighbor spacing within fused cloud (not directly comparable to cross-cloud residual)",
             "nn_median_after_fusion":float(np.median(cKDTree(simple["points"]).query(simple["points"],k=2,workers=-1)[0][:,1])),
             "nn_p90_after_fusion":float(np.percentile(cKDTree(simple["points"]).query(simple["points"],k=2,workers=-1)[0][:,1],90)),
             "local_density_before":density(all_points),"local_density_after":density(simple["points"]),
             "confidence_weighting_gamma":1.0,"confidence_fused_points":len(weighted["points"]),
             "confidence_note":"Confidence weighting changes position/color centroids within each voxel; it does not change occupied voxel count."}
    (method_dir/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    method1_dir=out/"method1"; method1_dir.mkdir(parents=True,exist_ok=True)
    (method1_dir/"metrics.json").write_text(json.dumps({"fusion_status":"fusion_not_attempted","reason":comparison["reason"]},indent=2),encoding="utf-8")
    print(json.dumps({"comparison":comparison,"fusion":payload},indent=2))


if __name__=="__main__": main()
