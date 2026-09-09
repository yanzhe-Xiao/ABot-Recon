#!/usr/bin/env python3
"""Deterministic headless PNG/MP4 renderer with one fixed camera setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from common import load_cloud_npz, transform_points, voxel_fuse

WIDTH, HEIGHT, FOV = 1280, 720, 50.0
SOURCE_COLOR = np.array([255, 105, 45], np.uint8)
TARGET_COLOR = np.array([0, 210, 255], np.uint8)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--target",type=Path,required=True)
    p.add_argument("--method1-dir",type=Path,required=True); p.add_argument("--method2-dir",type=Path,required=True)
    p.add_argument("--fusion-dir",type=Path,required=True); p.add_argument("--voxel",type=float,default=.05); return p.parse_args()


def sample(points: np.ndarray, colors: np.ndarray, limit: int=70000) -> tuple[np.ndarray,np.ndarray]:
    if len(points)<=limit: return points,colors
    keep=np.linspace(0,len(points)-1,limit,dtype=np.int64); return points[keep],colors[keep]


def camera(center: np.ndarray, radius: float, azimuth: float, elevation: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    az,el=np.radians([azimuth,elevation]); position=center+radius*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    forward=center-position; forward/=np.linalg.norm(forward); world_up=np.array([0.,0.,1.])
    if abs(forward@world_up)>.98: world_up=np.array([0.,1.,0.])
    right=np.cross(forward,world_up); right/=np.linalg.norm(right); up=np.cross(right,forward)
    return position,right,up,forward


def render(components: list[tuple[np.ndarray,np.ndarray]], center: np.ndarray, radius: float,
           azimuth: float, elevation: float, title: str, note: str="", radius_scale: float=1.0) -> np.ndarray:
    canvas=np.full((HEIGHT,WIDTH,3),(12,15,22),np.uint8); position,right,up,forward=camera(center,radius*radius_scale,azimuth,elevation)
    focal=.5*WIDTH/np.tan(np.radians(FOV/2)); projected=[]
    for points,colors in components:
        points,colors=sample(points,colors); delta=points-position; z=delta@forward; valid=z>max(radius*.01,.001)
        x=delta@right; y=delta@up; u=np.rint(focal*x[valid]/z[valid]+WIDTH/2).astype(int); v=np.rint(-focal*y[valid]/z[valid]+HEIGHT/2).astype(int)
        inside=(u>=0)&(u<WIDTH)&(v>=40)&(v<HEIGHT); projected.append((u[inside],v[inside],z[valid][inside],colors[valid][inside]))
    if projected:
        u=np.concatenate([x[0] for x in projected]); v=np.concatenate([x[1] for x in projected]); z=np.concatenate([x[2] for x in projected]); colors=np.concatenate([x[3] for x in projected])
        order=np.argsort(z)[::-1]; u,v,colors=u[order],v[order],colors[order]
        for du,dv in ((0,0),(1,0),(0,1),(1,1)):
            uu=np.clip(u+du,0,WIDTH-1); vv=np.clip(v+dv,0,HEIGHT-1); canvas[vv,uu]=colors
    cv2.putText(canvas,title,(24,34),cv2.FONT_HERSHEY_SIMPLEX,.78,(245,245,245),2,cv2.LINE_AA)
    if note: cv2.putText(canvas,note,(24,HEIGHT-24),cv2.FONT_HERSHEY_SIMPLEX,.6,(90,180,255),2,cv2.LINE_AA)
    return canvas


def video(path: Path, components: list[tuple[np.ndarray,np.ndarray]], setup: dict, title: str,
          note: str="", closeup: bool=False, overview: bool=False) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(path,fps=10,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="warning") as writer:
        for i in range(120):
            if overview: az=setup["azimuth_deg"]+20*np.sin(2*np.pi*i/120)
            else: az=setup["azimuth_deg"]+360*i/120
            writer.append_data(render(components,np.array(setup["center"]),setup["radius"],az,setup["elevation_deg"],title,note,.58 if closeup else 1.0))


def static_set(directory: Path, stages: dict[str,tuple[list[tuple[np.ndarray,np.ndarray]],str]], setup: dict) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    video_names={"before_alignment":"registration_before.mp4","after_coarse":"registration_coarse.mp4",
                 "after_refinement":"registration_refined.mp4","fusion_voxel":"fusion_voxel.mp4",
                 "fusion_confidence":"fusion_confidence.mp4"}
    for name,(components,note) in stages.items():
        image=render(components,np.array(setup["center"]),setup["radius"],setup["azimuth_deg"],setup["elevation_deg"],name.replace('_',' ').title(),note)
        imageio.imwrite(directory/f"{name}.png",image); video(directory/video_names[name],components,setup,name.replace('_',' ').title(),note)
    refined=stages["after_refinement"][0]
    views={"top_view":(setup["azimuth_deg"],88,1.0),"side_view":(setup["azimuth_deg"]+90,5,1.0),"overlap_closeup":(setup["azimuth_deg"],setup["elevation_deg"],.58),"c_to_d_view":(setup["azimuth_deg"]+180,12,1.0)}
    for name,(az,el,scale) in views.items(): imageio.imwrite(directory/f"{name}.png",render(refined,np.array(setup["center"]),setup["radius"],az,el,name.replace('_',' ').title(),"fixed shared camera setup",scale))


def main() -> None:
    a=parse_args(); src,tgt=load_cloud_npz(a.source),load_cloud_npz(a.target); p0,p8=src["points"],tgt["points"]
    pseudo_src=np.broadcast_to(SOURCE_COLOR,(len(p0),3)); pseudo_tgt=np.broadcast_to(TARGET_COLOR,(len(p8),3))
    m1c=np.load(a.method1_dir/"transforms/coarse_transform.npy"); m1f=np.load(a.method1_dir/"transforms/fine_transform.npy")
    m2c=np.load(a.method2_dir/"transforms/umeyama_sim3.npy"); m2f=np.load(a.method2_dir/"transforms/T_method2_final.npy")
    transforms=[np.eye(4),m1c,m1f,m2c,m2f]; bounds=np.concatenate([transform_points(p0,t) for t in transforms]+[p8]); low,high=np.percentile(bounds,[1,99],axis=0); center=(low+high)/2; radius=float(np.linalg.norm(high-low)*1.15)
    setup={"center":center.tolist(),"radius":radius,"azimuth_deg":-55.0,"elevation_deg":24.0,"fov_deg":FOV,"near":radius*.01,"far":radius*4,"resolution":[WIDTH,HEIGHT],"point_size_px":2,"background_rgb":[12,15,22],"coordinate_convention":"ABot world XYZ; Z used as renderer up","frames":120,"fps":10}
    root=a.method1_dir.parent; (root/"comparison").mkdir(parents=True,exist_ok=True); (root/"comparison/camera_setup.json").write_text(json.dumps(setup,indent=2),encoding="utf-8")
    before=[(p0,pseudo_src),(p8,pseudo_tgt)]
    m1coarse=[(transform_points(p0,m1c),pseudo_src),(p8,pseudo_tgt)]; m1fine=[(transform_points(p0,m1f),pseudo_src),(p8,pseudo_tgt)]
    failure="FUSION NOT ATTEMPTED: registration_geometrically_wrong"
    stages1={"before_alignment":(before,"06_clean=orange, 08=cyan"),"after_coarse":(m1coarse,"KISS-Matcher coarse"),"after_refinement":(m1fine,"small_gicp VGICP; rejected by feature correspondences"),"fusion_voxel":(m1fine,failure),"fusion_confidence":(m1fine,failure)}
    static_set(a.method1_dir/"renders",stages1,setup)
    m2coarse=[(transform_points(p0,m2c),pseudo_src),(p8,pseudo_tgt)]; m2fine=[(transform_points(p0,m2f),pseudo_src),(p8,pseudo_tgt)]
    aligned=transform_points(p0,m2f).astype(np.float32); allp=np.concatenate((aligned,p8)); allc=np.concatenate((src["colors"],tgt["colors"])); allw=np.concatenate((src["confidence"],tgt["confidence"]))
    simple=voxel_fuse(allp,allc,allw,a.voxel,False); weighted=voxel_fuse(allp,allc,allw,a.voxel,True)
    stages2={"before_alignment":(before,"06_clean=orange, 08=cyan"),"after_coarse":(m2coarse,"RANSAC Umeyama Sim(3)"),"after_refinement":(m2fine,"Sim(3) scale + small_gicp VGICP"),"fusion_voxel":([(simple["points"],simple["colors"])],"RGB equal-weight voxel fusion"),"fusion_confidence":([(weighted["points"],weighted["colors"])],"RGB confidence-weighted voxel fusion, gamma=1")}
    static_set(a.method2_dir/"renders",stages2,setup)
    for name,cloud in (("voxel",simple),("confidence",weighted)):
        components=[(cloud["points"],cloud["colors"])]; directory=a.fusion_dir/"renders"/name
        video(directory/"overview.mp4",components,setup,f"{name.title()} fusion overview",overview=True)
        video(directory/"orbit.mp4",components,setup,f"{name.title()} fusion orbit")
        video(directory/"overlap_closeup.mp4",components,setup,f"{name.title()} fusion overlap close-up",closeup=True)
    print(json.dumps(setup,indent=2))


if __name__=="__main__": main()
