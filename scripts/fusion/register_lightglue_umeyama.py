#!/usr/bin/env python3
"""Method 2: ALIKED retrieval + LightGlue + lifted 3D RANSAC/Umeyama + VGICP."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import small_gicp
import torch
from PIL import Image, ImageDraw
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd

from abot_recon.preprocessing import preprocess_image
from common import (load_cloud_npz, load_tensor, matrix, registration_metrics,
                    transform_points, umeyama, write_ply)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--source-video", type=Path, required=True)
    p.add_argument("--target-video", type=Path, required=True); p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--target-dir", type=Path, required=True); p.add_argument("--source-cloud", type=Path, required=True)
    p.add_argument("--target-cloud", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--keyframe-stride", type=int, default=30); p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--confidence-threshold", type=float, default=0.3); p.add_argument("--ransac-threshold", type=float, default=0.15)
    p.add_argument("--voxel", type=float, default=0.05); return p.parse_args()


def selected_frames(video: Path, indices: list[int]) -> dict[int, torch.Tensor]:
    wanted, output = set(indices), {}; capture = cv2.VideoCapture(str(video)); index = 0
    while wanted:
        ok, bgr = capture.read()
        if not ok: break
        if index in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            output[index] = preprocess_image(rgb, height=280, width=504)[0]
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted: raise ValueError(f"Could not decode selected frames {sorted(wanted)} from {video}")
    return output


def cpu_feature(feature: dict) -> dict:
    return {key: ([item.detach().cpu() for item in value] if isinstance(value, list) else value.detach().cpu()) for key, value in feature.items()}


def gpu_feature(feature: dict, device: str) -> dict:
    return {key: ([item.to(device) for item in value] if isinstance(value, list) else value.to(device)) for key, value in feature.items()}


def global_descriptor(feature: dict) -> np.ndarray:
    desc = rbd(feature)["descriptors"].float()
    value = torch.cat((desc.mean(0), desc.amax(0)))
    value = value / value.norm().clamp_min(1e-9)
    return value.numpy()


def draw_matches(image0: torch.Tensor, image1: torch.Tensor, p0: np.ndarray, p1: np.ndarray,
                 path: Path, limit: int = 250) -> None:
    left = (image0.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    right = (image1.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    canvas = Image.new("RGB", (1008, 280)); canvas.paste(Image.fromarray(left), (0, 0)); canvas.paste(Image.fromarray(right), (504, 0))
    draw = ImageDraw.Draw(canvas); rng = np.random.default_rng(19)
    chosen = np.arange(len(p0)) if len(p0) <= limit else rng.choice(len(p0), limit, replace=False)
    for idx in chosen:
        color = tuple(int(x) for x in rng.integers(60, 256, 3)); a = tuple(p0[idx]); b = (float(p1[idx, 0] + 504), float(p1[idx, 1]))
        draw.line((a, b), fill=color, width=1); draw.ellipse((a[0]-2,a[1]-2,a[0]+2,a[1]+2), fill=color); draw.ellipse((b[0]-2,b[1]-2,b[0]+2,b[1]+2), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def contact_sheet(rows: list[dict], src_images: dict[int, torch.Tensor], tgt_images: dict[int, torch.Tensor], path: Path) -> None:
    show = rows[:10]; sheet = Image.new("RGB", (1008, 280 * len(show)), "black"); draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(show):
        y = row_idx * 280
        for x, tensor in ((0, src_images[row["frame_06"]]), (504, tgt_images[row["frame_08"]])):
            array = (tensor.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8); sheet.paste(Image.fromarray(array), (x,y))
        draw.rectangle((0,y,1008,y+24), fill=(0,0,0)); draw.text((8,y+5), f"rank {row['rank']} | 06:{row['frame_06']}  08:{row['frame_08']}  score:{row['retrieval_score']:.4f}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)


def residuals(source: np.ndarray, target: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)


def ransac_umeyama(source: np.ndarray, target: np.ndarray, estimate_scale: bool, threshold: float,
                   iterations: int = 5000) -> dict[str, object]:
    rng = np.random.default_rng(20260908); best = np.zeros(len(source), dtype=bool); best_median = np.inf
    for _ in range(iterations):
        ids = rng.choice(len(source), 3, replace=False)
        try: scale, rotation, translation = umeyama(source[ids], target[ids], estimate_scale)
        except np.linalg.LinAlgError: continue
        if not np.isfinite(scale) or scale <= 0.05 or scale >= 10: continue
        err = residuals(source, target, scale, rotation, translation); inside = err <= threshold
        median = float(np.median(err[inside])) if inside.any() else np.inf
        if inside.sum() > best.sum() or (inside.sum() == best.sum() and median < best_median): best, best_median = inside, median
    if best.sum() < 3: raise ValueError("RANSAC found fewer than 3 inliers")
    scale, rotation, translation = umeyama(source[best], target[best], estimate_scale)
    err = residuals(source, target, scale, rotation, translation); best = err <= threshold
    scale, rotation, translation = umeyama(source[best], target[best], estimate_scale); err = residuals(source, target, scale, rotation, translation)
    return {"scale": float(scale), "rotation": rotation, "translation": translation, "inliers": best,
            "inlier_count": int(best.sum()), "inlier_ratio": float(best.mean()),
            "residual_median": float(np.median(err[best])), "residual_p90": float(np.percentile(err[best],90)),
            "raw_residual_median": float(np.median(err)), "raw_residual_p90": float(np.percentile(err,90))}


def save_similarity(path: Path, result: dict[str, object]) -> np.ndarray:
    rigid = matrix(result["rotation"], result["translation"]); similarity = rigid.copy(); similarity[:3,:3] *= result["scale"]
    np.save(path, similarity); np.savetxt(path.with_suffix(".txt"), similarity, fmt="%.10g"); return rigid


def main() -> None:
    args = parse_args(); out = args.output_dir
    candidates_dir, matches_dir, corr_dir, transforms_dir, clouds_dir = [out / name for name in ("frame_candidates","matches","correspondences","transforms","pointclouds")]
    for path in (candidates_dir,matches_dir,corr_dir,transforms_dir,clouds_dir): path.mkdir(parents=True, exist_ok=True)
    log=[]; metrics={"method":"ALIKED + LightGlue -> RANSAC Umeyama Sim(3)/SE(3) -> small_gicp VGICP","status":"started"}
    try:
        device="cuda" if torch.cuda.is_available() else "cpu"; src_ids=list(range(0,855,args.keyframe_stride)); tgt_ids=list(range(0,740,args.keyframe_stride))
        if src_ids[-1]!=854: src_ids.append(854)
        if tgt_ids[-1]!=739: tgt_ids.append(739)
        src_images=selected_frames(args.source_video,src_ids); tgt_images=selected_frames(args.target_video,tgt_ids)
        extractor=ALIKED(max_num_keypoints=2048).eval().to(device); matcher=LightGlue(features="aliked").eval().to(device)
        start=time.perf_counter(); src_feats={i:cpu_feature(extractor.extract(src_images[i].to(device),resize=None)) for i in src_ids}
        tgt_feats={i:cpu_feature(extractor.extract(tgt_images[i].to(device),resize=None)) for i in tgt_ids}
        src_desc=np.stack([global_descriptor(src_feats[i]) for i in src_ids]); tgt_desc=np.stack([global_descriptor(tgt_feats[i]) for i in tgt_ids])
        scores=src_desc@tgt_desc.T; flat=np.argsort(scores.ravel())[::-1]; rows=[]
        for index in flat:
            a,b=np.unravel_index(index,scores.shape); frame_a,frame_b=src_ids[a],tgt_ids[b]
            if any(abs(frame_a-r["frame_06"])<args.keyframe_stride and abs(frame_b-r["frame_08"])<args.keyframe_stride for r in rows): continue
            rows.append({"frame_06":frame_a,"timestamp_06":frame_a/(30000/1001),"frame_08":frame_b,"timestamp_08":frame_b/(30000/1001),"retrieval_score":float(scores[a,b]),"rank":len(rows)+1})
            if len(rows)>=args.top_k: break
        with (candidates_dir/"candidate_frame_pairs.csv").open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        contact_sheet(rows,src_images,tgt_images,candidates_dir/"candidate_contact_sheet.png")
        source_points=load_tensor(args.source_dir/"local_points.pt").float(); target_points=load_tensor(args.target_dir/"local_points.pt").float()
        source_conf=load_tensor(args.source_dir/"confidence.pt").float(); target_conf=load_tensor(args.target_dir/"confidence.pt").float()
        source_poses=np.load(args.source_dir/"camera_poses.npy"); target_poses=np.load(args.target_dir/"camera_poses.npy")
        all_src=[]; all_tgt=[]; all_score=[]; pair_stats=[]; raw_matches=0; valid_matches=0
        for row in rows:
            start_pair=time.perf_counter(); f0=gpu_feature(src_feats[row["frame_06"]],device); f1=gpu_feature(tgt_feats[row["frame_08"]],device)
            match_out=matcher({"image0":f0,"image1":f1}); f0u,f1u,mu=[rbd(x) for x in (f0,f1,match_out)]
            pairs=mu["matches"].detach().cpu(); scores_m=mu["scores"].detach().cpu().numpy(); p0=f0u["keypoints"][pairs[:,0]].detach().cpu().numpy(); p1=f1u["keypoints"][pairs[:,1]].detach().cpu().numpy()
            draw_matches(src_images[row["frame_06"]],tgt_images[row["frame_08"]],p0,p1,matches_dir/f"pair_{row['rank']:03d}_06_{row['frame_06']:04d}_08_{row['frame_08']:04d}.png")
            uv0=np.rint(p0).astype(int); uv1=np.rint(p1).astype(int); uv0[:,0]=np.clip(uv0[:,0],0,503); uv0[:,1]=np.clip(uv0[:,1],0,279); uv1[:,0]=np.clip(uv1[:,0],0,503); uv1[:,1]=np.clip(uv1[:,1],0,279)
            lp0=source_points[row["frame_06"],uv0[:,1],uv0[:,0]].numpy(); lp1=target_points[row["frame_08"],uv1[:,1],uv1[:,0]].numpy()
            cf0=source_conf[row["frame_06"],uv0[:,1],uv0[:,0]].numpy(); cf1=target_conf[row["frame_08"],uv1[:,1],uv1[:,0]].numpy()
            valid=np.isfinite(lp0).all(1)&np.isfinite(lp1).all(1)&(lp0[:,2]>0)&(lp1[:,2]>0)&(cf0>=args.confidence_threshold)&(cf1>=args.confidence_threshold)
            wp0=lp0[valid]@source_poses[row["frame_06"],:3,:3].T+source_poses[row["frame_06"],:3,3]; wp1=lp1[valid]@target_poses[row["frame_08"],:3,:3].T+target_poses[row["frame_08"],:3,3]
            all_src.append(wp0); all_tgt.append(wp1); all_score.append(scores_m[valid]); raw_matches+=len(p0); valid_matches+=int(valid.sum())
            pair_stats.append({**row,"keypoints_06":len(f0u["keypoints"]),"keypoints_08":len(f1u["keypoints"]),"match_count":len(p0),"valid_3d_matches":int(valid.sum()),"valid_ratio":float(valid.mean()) if len(valid) else 0.0,"match_confidence_mean":float(scores_m.mean()) if len(scores_m) else 0.0,"runtime_sec":time.perf_counter()-start_pair})
        del source_points,target_points,source_conf,target_conf
        p06=np.concatenate(all_src); p08=np.concatenate(all_tgt); match_scores=np.concatenate(all_score)
        np.savez_compressed(corr_dir/"correspondences_3d.npz",points_06=p06,points_08=p08,match_score=match_scores)
        (matches_dir/"pair_metrics.json").write_text(json.dumps(pair_stats,indent=2),encoding="utf-8")
        metrics["retrieval"]={"keyframes_06":len(src_ids),"keyframes_08":len(tgt_ids),"candidate_pairs":len(rows),"runtime_including_features_sec":time.perf_counter()-start}
        metrics["correspondences"]={"raw_2d_matches":raw_matches,"valid_3d_matches":valid_matches,"confidence_filtered_matches":raw_matches-valid_matches,"valid_ratio":valid_matches/max(raw_matches,1)}
        start_r=time.perf_counter(); se3=ransac_umeyama(p06,p08,False,args.ransac_threshold); sim3=ransac_umeyama(p06,p08,True,args.ransac_threshold); ransac_runtime=time.perf_counter()-start_r
        se3_rigid=save_similarity(transforms_dir/"umeyama_se3.npy",se3); sim3_rigid=save_similarity(transforms_dir/"umeyama_sim3.npy",sim3)
        metrics["umeyama_se3"]={k:v for k,v in se3.items() if k not in ("rotation","translation","inliers")}
        metrics["umeyama_sim3"]={k:v for k,v in sim3.items() if k not in ("rotation","translation","inliers")}; metrics["runtime_ransac_sec"]=ransac_runtime
        src_cloud,tgt_cloud=load_cloud_npz(args.source_cloud),load_cloud_npz(args.target_cloud)
        initial_scaled=src_cloud["points"].astype(np.float64)*sim3["scale"]; start_ref=time.perf_counter()
        refined=small_gicp.align(tgt_cloud["points"].astype(np.float64),initial_scaled,init_T_target_source=sim3_rigid,registration_type="VGICP",voxel_resolution=args.voxel,downsampling_resolution=args.voxel,max_correspondence_distance=args.ransac_threshold,num_threads=8,max_iterations=50)
        refine_runtime=time.perf_counter()-start_ref; final_rigid=np.asarray(refined.T_target_source); final_similarity=final_rigid.copy(); final_similarity[:3,:3]*=sim3["scale"]
        np.save(transforms_dir/"T_method2_final.npy",final_similarity); np.savetxt(transforms_dir/"T_method2_final.txt",final_similarity,fmt="%.10g")
        before=registration_metrics(src_cloud["points"],tgt_cloud["points"],np.eye(4),1.0,args.ransac_threshold)
        sim_metric=registration_metrics(src_cloud["points"],tgt_cloud["points"],sim3_rigid,sim3["scale"],args.ransac_threshold)
        final_metric=registration_metrics(src_cloud["points"],tgt_cloud["points"],final_rigid,sim3["scale"],args.ransac_threshold)
        final_metric.update({"runtime_sec":refine_runtime,"converged":bool(refined.converged),"iterations":int(refined.iterations),"library_num_inliers":int(refined.num_inliers),"library_error":float(refined.error)})
        metrics.update({"before":before,"sim3_initial":sim_metric,"fine":final_metric,"coarse_status":"success" if sim3["inlier_count"]>=10 else "2d_to_3d_failed","refine_status":"success" if refined.converged else "gicp_failed"})
        plausible=(sim3["inlier_count"]>=10 and 0.4<=sim3["scale"]<=2.5 and refined.converged and final_metric["fitness"]>=0.2 and final_metric["overlap_nn_median"]<=0.3)
        metrics["status"]="likely_success" if plausible else "uncertain"; metrics["runtime_total_sec"]=metrics["retrieval"]["runtime_including_features_sec"]+ransac_runtime+refine_runtime
        aligned=transform_points(src_cloud["points"],final_rigid,sim3["scale"]); write_ply(clouds_dir/"06_clean_refined_in_08.ply",aligned,src_cloud["colors"],src_cloud["confidence"])
    except Exception:
        metrics["status"]="2d_match_failed" if "correspondences" not in metrics else "2d_to_3d_failed"; metrics["error"]=traceback.format_exc(); log.append(metrics["error"])
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); (out/"run.log").write_text("\n".join(log)+"\n"+json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
