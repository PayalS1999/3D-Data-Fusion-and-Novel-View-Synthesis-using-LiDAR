#!/usr/bin/env python3
"""
prepare_kitti_nerf_depth.py

Single script to prepare a KITTI-00 style scene for Nerfstudio (nerfacto / depth-nerfacto):

Creates:
  - images/*.png          (RGB frames from camera 2)
  - depth/*.png           (uint16 depth in millimetres, rectified cam2)
  - transforms.json       (intrinsics + poses + split + depth_file_path)

Splits:
  - Every K-th *processed* frame is labeled "val"
  - Others are labeled "train"

Example usage:

  python prepare_kitti_nerf_depth.py \
      --kitti_root /home/.../KITTI/dataset \
      --sequence 00 \
      --raw_calib_dir /home/.../KITTI/2011_10_03 \
      --poses_file /home/.../KITTI/dataset/poses/00.txt \
      --out_dir /home/.../kitti_nerf_scene \
      --start 0 --end 500 --step 1 \
      --val_every 15
"""

import os
import json
import argparse
import shutil
import numpy as np
import cv2

# ----------------- Basic utilities ----------------- #

def parse_keyvals(path, keys_keep=None):
    out = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, vals = line.split(":", 1)
            k = k.strip()
            if keys_keep and k not in keys_keep:
                continue
            nums = [float(x) for x in vals.strip().split()]
            out[k] = np.array(nums, dtype=np.float64)
    return out


def load_raw_calib(raw_dir):
    """
    KITTI Raw calib:
      - calib_cam_to_cam.txt   => P_rect_02, R_rect_00 / R0_rect
      - calib_velo_to_cam.txt  => R, T

    Returns:
      P2             (3x4)   : rectified projection for camera 2
      T_cam0_from_velo (4x4) : rectified cam0 from Velodyne
      T_cam2_from_cam0 (4x4) : rectified cam2 from rectified cam0 (pure baseline)
    """
    f_cam  = os.path.join(raw_dir, "calib_cam_to_cam.txt")
    f_velo = os.path.join(raw_dir, "calib_velo_to_cam.txt")

    cam = parse_keyvals(
        f_cam, keys_keep={"P_rect_02", "R_rect_00", "R0_rect"}
    )
    P2 = cam["P_rect_02"].reshape(3, 4)

    if "R_rect_00" in cam:
        Rrect = cam["R_rect_00"].reshape(3, 3)
    else:
        Rrect = cam["R0_rect"].reshape(3, 3)

    R_rect_00_4 = np.eye(4, dtype=np.float64)
    R_rect_00_4[:3, :3] = Rrect

    velo = parse_keyvals(f_velo, keys_keep={"R", "T"})
    R = velo["R"].reshape(3, 3)
    T = velo["T"].reshape(3)

    Tr_velo_to_cam = np.eye(4, dtype=np.float64)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3]  = T

    # LiDAR → rectified cam0
    T_cam0_from_velo = R_rect_00_4 @ Tr_velo_to_cam

    # Cam0 -> Cam2 baseline from P2
    fx = P2[0, 0]
    Tx = P2[0, 3]
    baseline = -Tx / fx

    T_cam2_from_cam0 = np.eye(4, dtype=np.float64)
    T_cam2_from_cam0[0, 3] = baseline

    return P2, T_cam0_from_velo, T_cam2_from_cam0


def load_velodyne_bin(bin_path):
    """Load KITTI velodyne .bin (x,y,z,reflectance) → Nx3 xyz."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # xyz only


def make_homog(pts_xyz):
    """Nx3 → 4xN homogeneous."""
    pts_xyz = np.asarray(pts_xyz)
    N = pts_xyz.shape[0]
    out = np.ones((4, N), dtype=np.float64)
    out[:3, :] = pts_xyz.T
    return out


def load_poses(poses_file):
    """Return list of 4x4 camera-0 poses: T_world_from_cam0[i]."""
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            assert len(vals) == 12
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def project_points(P, X_cam4):
    """
    P: 3x4 projection matrix
    X_cam4: 4xN points in that camera frame
    Returns:
      u, v, z, mask (z>0)
    """
    uvw = P @ X_cam4  # (3, N)
    z   = uvw[2, :]
    mask = z > 1e-6
    u = uvw[0, mask] / z[mask]
    v = uvw[1, mask] / z[mask]
    return u, v, z[mask], mask


def forward_splat_depth(u, v, z, W, H, max_depth=80.0):
    """
    Simple z-buffer depth splat:
      - keeps nearest valid z per pixel
      - 0 indicates no depth
    u, v: float pixel coords
    z   : depth (meters)
    """
    depth = np.zeros((H, W), dtype=np.float32)  # 0 = no data

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    for x, y, d in zip(ui, vi, z):
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if d <= 0 or d > max_depth:
            continue
        if depth[y, x] == 0 or d < depth[y, x]:
            depth[y, x] = d

    return depth


# ----------------- Main script ----------------- #

def main():
    ap = argparse.ArgumentParser("Prepare KITTI scene for nerfacto / depth-nerfacto")
    ap.add_argument("--kitti_root", default= "/home/sriramg/payalsaha/KITTI/dataset/",
                    help="Root of KITTI odometry dataset, e.g. /home/.../KITTI/dataset")
    ap.add_argument("--sequence", default="00",
                    help="Sequence id, e.g. 00")
    ap.add_argument("--raw_calib_dir", default="/home/sriramg/payalsaha/KITTI/2011_10_03",
                    help="Raw calib date folder, e.g. /home/.../KITTI/2011_10_03")
    ap.add_argument("--poses_file", default="/home/sriramg/payalsaha/KITTI/dataset/poses/00.txt",
                    help="Poses file, e.g. /home/.../KITTI/dataset/poses/00.txt")
    ap.add_argument("--out_dir", default="/home/sriramg/payalsaha/kitti_nerf_500_s8",
                    help="Output scene root, e.g. /home/.../kitti_nerf_scene")
    ap.add_argument("--start", type=int, default=0,
                    help="First frame index (inclusive)")
    ap.add_argument("--end", type=int, default=4000,
                    help="Last frame index (exclusive)")
    ap.add_argument("--step", type=int, default=8,
                    help="Frame step")
    ap.add_argument("--val_every", type=int, default=20,
                    help="Every K-th processed frame is labeled 'val'")
    args = ap.parse_args()

    seq_dir  = os.path.join(args.kitti_root, "sequences", args.sequence)
    img_dir  = os.path.join(seq_dir, "image_2")
    velo_dir = os.path.join(seq_dir, "velodyne")

    # Output dirs
    os.makedirs(args.out_dir, exist_ok=True)
    out_img_dir   = os.path.join(args.out_dir, "images")
    out_depth_dir = os.path.join(args.out_dir, "depth")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    # Load calib + poses
    P2, T_cam0_from_velo, T_cam2_from_cam0 = load_raw_calib(args.raw_calib_dir)
    poses = load_poses(args.poses_file)

    # Intrinsics from P2
    fx = float(P2[0, 0])
    fy = float(P2[1, 1])
    cx = float(P2[0, 2])
    cy = float(P2[1, 2])

    # Image size from first available frame
    first_img = None
    for idx in range(args.start, args.end, args.step):
        test_path = os.path.join(img_dir, f"{idx:06d}.png")
        if os.path.isfile(test_path):
            first_img = cv2.imread(test_path)
            break

    if first_img is None:
        raise RuntimeError("No images found in the given range.")

    H, W = first_img.shape[:2]

    frames_json = []
    processed_count = 0

    for idx in range(args.start, args.end, args.step):
        img_path  = os.path.join(img_dir, f"{idx:06d}.png")
        velo_path = os.path.join(velo_dir, f"{idx:06d}.bin")

        if (not os.path.isfile(img_path)) or (not os.path.isfile(velo_path)):
            print(f"[WARN] Missing image or velodyne for frame {idx:06d}, skipping.")
            continue

        print(f"[INFO] Processing frame {idx:06d}...")

        # ----- 1) Copy image -----
        out_img_path = os.path.join(out_img_dir, f"{idx:06d}.png")
        if not os.path.isfile(out_img_path):
            shutil.copy(img_path, out_img_path)

        # ----- 2) Compute LiDAR depth in rectified cam2 -----
        pts_xyz = load_velodyne_bin(velo_path)
        X_velo  = make_homog(pts_xyz)                # 4xN

        # LiDAR → rectified cam0 → rectified cam2
        X_cam0 = T_cam0_from_velo @ X_velo           # 4xN
        X_cam2 = T_cam2_from_cam0 @ X_cam0           # 4xN

        u, v, z, _ = project_points(P2, X_cam2)

        depth_m = forward_splat_depth(u, v, z, W, H, max_depth=80.0)

        # Convert to millimetres for depth-nerfacto (uint16)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0
        depth_mm = (depth_m * 1000.0).round().astype(np.uint16)

        out_depth_path = os.path.join(out_depth_dir, f"{idx:06d}.png")
        cv2.imwrite(out_depth_path, depth_mm)

        # ----- 3) Pose: cam2 → world -----
        T_w_from_cam0   = poses[idx]             # world from cam0
        T_cam0_from_cam2 = np.linalg.inv(T_cam2_from_cam0)
        T_w_from_cam2   = T_w_from_cam0 @ T_cam0_from_cam2

        # ----- 4) Split assignment -----
        split = "train"
        if args.val_every > 0:
            if processed_count % args.val_every == 0:
                split = "val"
        processed_count += 1

        frame_entry = {
            "file_path":      f"images/{idx:06d}.png",
            "transform_matrix": T_w_from_cam2.tolist(),
            "split":          split,
            # Optional depth path (useful for custom loaders or DS-NeRF-style code)
            "depth_file_path": f"depth/{idx:06d}.png",
        }
        frames_json.append(frame_entry)

    # ----- 5) Global transforms.json -----
    camera_angle_x = float(2.0 * np.arctan(W / (2.0 * fx)))

    transforms = {
        "camera_angle_x": camera_angle_x,
        "fl_x": fx,
        "fl_y": fy,
        "cx":  cx,
        "cy":  cy,
        "w":   W,
        "h":   H,
        "frames": frames_json,
    }

    out_json = os.path.join(args.out_dir, "transforms.json")
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"[OK] Wrote {out_json} with {len(frames_json)} frames.")
    print(f"[OK] Images in: {out_img_dir}")
    print(f"[OK] Depth PNGs in: {out_depth_dir}")


if __name__ == "__main__":
    main()