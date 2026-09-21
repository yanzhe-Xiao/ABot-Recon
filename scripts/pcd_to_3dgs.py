#!/usr/bin/env python3
# ==============================================================================
# Point Cloud to 3D Gaussian Splatting (3DGS) Surfel Converter
# ==============================================================================
# Converts dense colored point clouds (with surface normals) directly into
# standard 3D Gaussian Splatting (3DGS) PLY format and .splat format.
#
# Key Features:
#   - Closed-form Gaussian Surfel Mapping:
#       * Position: (x, y, z) directly from point cloud
#       * Rotation: Quaternion (rot_0..3 = w, x, y, z) aligning local Z-axis with normal n
#       * Scale: Tangent scales (scale_0, scale_1 = ln(s_tangent)), thin normal (scale_2 = ln(s_normal))
#       * Opacity: Logit(alpha) = ln(alpha / (1 - alpha))
#       * Base Color: Spherical Harmonics Degree 0 DC coefficients f_dc_0..2
#   - High Efficiency:
#       * Fully vectorized NumPy arithmetic (zero training time, zero VRAM)
#       * Millions of points converted in under 1 second
#   - Universal Compatibility:
#       * Standard Inria 3DGS PLY (Degree 0 or Degree 3)
#       * Optional 32-byte binary .splat export for instant WebGL viewer playback
# ==============================================================================

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of rotation quaternion (w, x, y, z) that rotates
    local reference axis v1 = [0, 0, 1] to the target surface normal vector n.

    Args:
        normals: (N, 3) float array of unit normal vectors.

    Returns:
        quats: (N, 4) float32 array where each row is (rot_0=w, rot_1=x, rot_2=y, rot_3=z).
    """
    n = len(normals)
    quats = np.zeros((n, 4), dtype=np.float32)

    # Normalize normals to safeguard against unnormalized inputs
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    u_norm = normals / norms

    nx = u_norm[:, 0]
    ny = u_norm[:, 1]
    nz = u_norm[:, 2]

    # Standard half-angle quaternion for rotation from [0, 0, 1] to n:
    # Axis = [0, 0, 1] x n = [-ny, nx, 0]
    # Cos = [0, 0, 1] . n = nz
    # Unnormalized: w = 1 + nz, x = -ny, y = nx, z = 0
    w = 1.0 + nz
    x = -ny
    y = nx
    z = np.zeros_like(w)

    # Identify edge case where normal points directly opposite: nz ≈ -1
    opposite_mask = nz < -0.9999
    # For opposite vector [0, 0, -1], rotate 180 deg around X axis: q = (0, 1, 0, 0)
    w[opposite_mask] = 0.0
    x[opposite_mask] = 1.0
    y[opposite_mask] = 0.0
    z[opposite_mask] = 0.0

    # Stack and normalize
    q_unnorm = np.stack([w, x, y, z], axis=1)
    q_len = np.linalg.norm(q_unnorm, axis=1, keepdims=True)
    q_len[q_len < 1e-12] = 1.0
    quats = (q_unnorm / q_len).astype(np.float32)

    # Fallback for invalid/NaN normals: identity quaternion (1, 0, 0, 0)
    invalid_mask = ~np.isfinite(quats).all(axis=1)
    if np.any(invalid_mask):
        quats[invalid_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return quats


def compute_gaussian_scales(
    pts: np.ndarray,
    scale_mode: str = "adaptive",
    base_scale: float = 0.005,
    thin_factor: float = 0.15,
) -> np.ndarray:
    """
    Compute log-scales (scale_0, scale_1, scale_2) for anisotropic Gaussian surfels.

    Args:
        pts: (N, 3) float array of 3D point coordinates.
        scale_mode: "adaptive" (k-NN distance based) or "fixed" (base_scale based).
        base_scale: Default tangent radius in meters (e.g. 0.005m = 5mm).
        thin_factor: Ratio of normal thickness to tangent radius (default: 0.15).

    Returns:
        log_scales: (N, 3) float32 array representing [ln(s_x), ln(s_y), ln(s_z)].
    """
    n = len(pts)

    if scale_mode == "adaptive" and n > 10:
        from scipy.spatial import cKDTree

        k_query = min(4, n)
        tree = cKDTree(pts)
        if n > 2_000_000:
            chunk_size = 500_000
            mean_dists = np.empty(n, dtype=np.float32)
            for i in range(0, n, chunk_size):
                d_chunk, _ = tree.query(pts[i:i + chunk_size], k=k_query, workers=-1)
                mean_dists[i:i + chunk_size] = np.mean(d_chunk[:, 1:], axis=1)
        else:
            dists, _ = tree.query(pts, k=k_query, workers=-1)
            mean_dists = np.mean(dists[:, 1:], axis=1)

        # Surfel tangent radius based on local spacing, clamped to realistic bounds (0.5mm ~ 15mm)
        s_tangent = np.clip(mean_dists * 1.1, 0.0005, 0.015).astype(np.float32)
    else:
        s_tangent = np.full(n, base_scale, dtype=np.float32)

    s_normal = np.maximum(s_tangent * thin_factor, 0.0002).astype(np.float32)

    log_s0 = np.log(s_tangent)
    log_s1 = np.log(s_tangent)
    log_s2 = np.log(s_normal)

    return np.stack([log_s0, log_s1, log_s2], axis=1).astype(np.float32)


def write_3dgs_ply(
    filepath: Path | str,
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    rot_quats: np.ndarray,
    sh_degree: int = 0,
) -> int:
    """
    Write 3D Gaussian Splatting data directly into standard binary_little_endian PLY file.

    Args:
        filepath: Target output .ply path.
        xyz: (N, 3) float32 positions.
        normals: (N, 3) float32 normals.
        f_dc: (N, 3) float32 DC spherical harmonics colors.
        opacity_logit: (N,) float32 opacity values in logit space.
        log_scales: (N, 3) float32 log-scale parameters.
        rot_quats: (N, 4) float32 normalized quaternions (w, x, y, z).
        sh_degree: Spherical Harmonics degree (0 for base color only, 3 for 45 zero-padded f_rest_*).

    Returns:
        File size in bytes.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)

    # Construct dtype fields matching 3DGS PLY format
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    num_f_rest = 0
    if sh_degree == 1:
        num_f_rest = 9
    elif sh_degree == 2:
        num_f_rest = 24
    elif sh_degree == 3:
        num_f_rest = 45

    for i in range(num_f_rest):
        fields.append((f"f_rest_{i}", "f4"))

    fields.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ])

    vertex_dtype = np.dtype(fields)
    vertex_data = np.empty(n, dtype=vertex_dtype)

    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]

    vertex_data["nx"] = normals[:, 0]
    vertex_data["ny"] = normals[:, 1]
    vertex_data["nz"] = normals[:, 2]

    vertex_data["f_dc_0"] = f_dc[:, 0]
    vertex_data["f_dc_1"] = f_dc[:, 1]
    vertex_data["f_dc_2"] = f_dc[:, 2]

    for i in range(num_f_rest):
        vertex_data[f"f_rest_{i}"] = 0.0

    vertex_data["opacity"] = opacity_logit
    vertex_data["scale_0"] = log_scales[:, 0]
    vertex_data["scale_1"] = log_scales[:, 1]
    vertex_data["scale_2"] = log_scales[:, 2]

    vertex_data["rot_0"] = rot_quats[:, 0]  # w
    vertex_data["rot_1"] = rot_quats[:, 1]  # x
    vertex_data["rot_2"] = rot_quats[:, 2]  # y
    vertex_data["rot_3"] = rot_quats[:, 3]  # z

    # Build ASCII Header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for name, dtype_str in fields:
        header_lines.append(f"property float {name}")
    header_lines.append("end_header\n")
    header_bytes = "\n".join(header_lines).encode("ascii")

    # Write header and binary payload
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(vertex_data.tobytes())

    return filepath.stat().st_size


def write_splat_binary(
    filepath: Path | str,
    xyz: np.ndarray,
    rgb_uint8: np.ndarray,
    opacity_alpha: np.ndarray,
    scales_m: np.ndarray,
    rot_quats: np.ndarray,
) -> int:
    """
    Write compact 32-byte per Gaussian .splat binary format (antimatter15 WebGL splat standard):
        - position: 3 x float32 (12 bytes)
        - scale:    3 x float32 (12 bytes)
        - color:    4 x uint8 (RGBA, 4 bytes)
        - rotation: 4 x uint8 (quantized quaternion, 4 bytes: 128 + 128 * q)
    Total: 32 bytes per splat.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)

    splat_dtype = np.dtype([
        ("pos", "3f4"),
        ("scale", "3f4"),
        ("color", "4u1"),
        ("rot", "4u1"),
    ])

    data = np.empty(n, dtype=splat_dtype)
    data["pos"] = xyz.astype(np.float32)
    data["scale"] = scales_m.astype(np.float32)

    # Color RGBA
    alpha_uint8 = np.clip(opacity_alpha * 255.0, 0, 255).astype(np.uint8)
    data["color"][:, :3] = rgb_uint8
    data["color"][:, 3] = alpha_uint8

    # Quantize quaternion (rot_0..3 = w, x, y, z) into uint8 [0, 255]
    q_quant = np.clip(np.round(128.0 + 128.0 * rot_quats), 0, 255).astype(np.uint8)
    data["rot"] = q_quant

    with open(filepath, "wb") as f:
        f.write(data.tobytes())

    return filepath.stat().st_size


def convert_point_cloud_to_3dgs(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    voxel_size: Optional[float] = None,
    scale_mode: str = "adaptive",
    base_scale: float = 0.006,
    thin_factor: float = 0.15,
    opacity: float = 0.95,
    sh_degree: int = 0,
    export_splat: bool = True,
    invert_z: bool = False,
) -> dict:
    """
    Convert a point cloud into a 3D Gaussian Splatting representation.

    Args:
        input_path: Path to input point cloud (.ply, .pcd, etc.).
        output_path: Path to output 3DGS PLY file.
        voxel_size: Optional spatial downsample voxel size (meters) before conversion.
        scale_mode: "adaptive" or "fixed".
        base_scale: Surfel tangent radius in meters.
        thin_factor: Ratio of surfel normal thickness to tangent radius.
        opacity: Base opacity alpha in [0, 1].
        sh_degree: 0 (minimal size) or 3 (legacy compatible).
        export_splat: Whether to also export antimatter15 .splat binary file.
        invert_z: Whether to invert Z-axis coordinates and normal vectors.

    Returns:
        Report dictionary with conversion statistics.
    """
    t0 = time.perf_counter()
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {input_path}")

    if output_path is None:
        out_ply_path = input_path.parent / f"{input_path.stem}_3dgs.ply"
    else:
        out_ply_path = Path(output_path)

    print(f"\n{'='*75}")
    print("3D Gaussian Splatting (3DGS) Surfel Converter")
    print(f"{'='*75}")
    print(f"Input Point Cloud:  {input_path}")
    print(f"Output 3DGS PLY:    {out_ply_path}")
    print(f"Scale Mode:         {scale_mode} (base_scale={base_scale}m, thin_factor={thin_factor})")
    print(f"Opacity:            {opacity} (logit={math.log(opacity / (1.0 - opacity)):.3f})")
    print(f"SH Degree:          {sh_degree}")
    print(f"{'='*75}\n")

    # 1. Load Point Cloud
    print("[1/5] Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_path))
    n_raw = len(pcd.points)
    if n_raw == 0:
        raise ValueError(f"Input point cloud contains 0 points: {input_path}")
    print(f"  Loaded {n_raw:,} points from disk.")

    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        print(f"  Downsampling to voxel size {voxel_size}m...")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  Voxel downsampled: {n_raw:,} -> {len(pcd.points):,} points.")

    pts = np.asarray(pcd.points, dtype=np.float32)
    n_pts = len(pts)

    if invert_z:
        print("  Inverting Z-axis coordinates (Z -> -Z)...")
        pts[:, 2] = -pts[:, 2]

    # 2. Extract or Estimate Normals
    print("[2/5] Resolving surface normals...")
    if not pcd.has_normals() or len(pcd.normals) != n_pts:
        print("  Normals missing or incomplete. Estimating normals via KD-Tree...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        center = np.mean(pts, axis=0)
        pcd.orient_normals_towards_camera_location(center + np.array([0.0, 2.0, 0.0], dtype=np.float32))
        print("  Normals estimated and oriented successfully.")
    normals = np.asarray(pcd.normals, dtype=np.float32)
    if invert_z:
        normals[:, 2] = -normals[:, 2]

    # 3. Extract and Transform Colors to Spherical Harmonics (Degree 0 DC)
    print("[3/5] Computing Spherical Harmonics (Degree 0 DC colors)...")
    if pcd.has_colors() and len(pcd.colors) == n_pts:
        colors_rgb = np.asarray(pcd.colors, dtype=np.float32)  # in [0, 1]
    else:
        print("  Colors missing. Defaulting to neutral light gray.")
        colors_rgb = np.full((n_pts, 3), 0.75, dtype=np.float32)

    # Clamp colors to [0, 1]
    colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
    # Convert RGB to SH DC: f_dc = (RGB - 0.5) / SH_C0
    f_dc = ((colors_rgb - 0.5) / SH_C0).astype(np.float32)

    # 4. Closed-form Surfel-to-Gaussian Parameter Mapping
    print("[4/5] Computing closed-form Gaussian Surfel parameters (quaternions & scales)...")
    t_map0 = time.perf_counter()

    # Quaternions: align local Z with surface normal
    rot_quats = normal_to_quaternion(normals)

    # Scales: anisotropic surfel scales in log space
    log_scales = compute_gaussian_scales(
        pts, scale_mode=scale_mode, base_scale=base_scale, thin_factor=thin_factor
    )

    # Opacity in logit space
    opacity_clamped = np.clip(opacity, 1e-4, 1.0 - 1e-4)
    opacity_logit_scalar = float(math.log(opacity_clamped / (1.0 - opacity_clamped)))
    opacity_logits = np.full(n_pts, opacity_logit_scalar, dtype=np.float32)
    print(f"  Gaussian parameters computed in {time.perf_counter() - t_map0:.3f}s.")

    # 5. Export Deliverables
    print("[5/5] Exporting 3D Gaussian Splatting deliverables...")
    t_exp0 = time.perf_counter()
    ply_size = write_3dgs_ply(
        filepath=out_ply_path,
        xyz=pts,
        normals=normals,
        f_dc=f_dc,
        opacity_logit=opacity_logits,
        log_scales=log_scales,
        rot_quats=rot_quats,
        sh_degree=sh_degree,
    )
    ply_mb = round(ply_size / 1024 / 1024, 2)
    print(f"  [Exported 3DGS PLY]: {out_ply_path} ({ply_mb} MB in {time.perf_counter() - t_exp0:.2f}s)")

    splat_path = None
    splat_mb = 0.0
    if export_splat:
        t_splat0 = time.perf_counter()
        out_splat_path = out_ply_path.with_suffix(".splat")
        scales_m = np.exp(log_scales)
        rgb_uint8 = (colors_rgb * 255.0).astype(np.uint8)
        opacity_alphas = np.full(n_pts, opacity_clamped, dtype=np.float32)

        splat_size = write_splat_binary(
            filepath=out_splat_path,
            xyz=pts,
            rgb_uint8=rgb_uint8,
            opacity_alpha=opacity_alphas,
            scales_m=scales_m,
            rot_quats=rot_quats,
        )
        splat_mb = round(splat_size / 1024 / 1024, 2)
        splat_path = str(out_splat_path)
        print(f"  [Exported .splat]:   {out_splat_path} ({splat_mb} MB in {time.perf_counter() - t_splat0:.2f}s)")

    total_time = round(time.perf_counter() - t0, 2)
    print(f"\n{'='*75}")
    print(f"Conversion Completed Successfully in {total_time}s!")
    print(f"Total Gaussians: {n_pts:,}")
    print(f"{'='*75}\n")

    return {
        "status": "success",
        "input_point_cloud": str(input_path),
        "output_3dgs_ply": str(out_ply_path),
        "output_splat": splat_path,
        "total_gaussians": n_pts,
        "ply_size_mb": ply_mb,
        "splat_size_mb": splat_mb,
        "conversion_time_seconds": total_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert dense colored point cloud to 3D Gaussian Splatting (3DGS) PLY format."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to input point cloud PLY"
    )
    parser.add_argument(
        "-o", "--output", default=None, type=str, help="Path to output 3DGS PLY file"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=None, help="Optional voxel downsample size in meters"
    )
    parser.add_argument(
        "--scale-mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Scale computation mode: adaptive (k-NN based) or fixed",
    )
    parser.add_argument(
        "--base-scale", type=float, default=0.006, help="Base surfel radius in meters (default: 0.006)"
    )
    parser.add_argument(
        "--thin-factor", type=float, default=0.15, help="Ratio of normal thickness to tangent radius"
    )
    parser.add_argument(
        "--opacity", type=float, default=0.95, help="Default opacity alpha in [0, 1] (default: 0.95)"
    )
    parser.add_argument(
        "--sh-degree", type=int, choices=[0, 3], default=0, help="Spherical Harmonics degree (0 or 3)"
    )
    parser.add_argument(
        "--no-splat", action="store_true", help="Do not generate companion .splat file"
    )
    parser.add_argument(
        "--invert-z", action="store_true", help="Invert Z coordinates and flip normal Z before conversion"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_point_cloud_to_3dgs(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        scale_mode=args.scale_mode,
        base_scale=args.base_scale,
        thin_factor=args.thin_factor,
        opacity=args.opacity,
        sh_degree=args.sh_degree,
        export_splat=not args.no_splat,
        invert_z=args.invert_z,
    )


if __name__ == "__main__":
    main()
