#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2

# ------------------ Basic utils ------------------ #

def load_velodyne_bin(bin_path):
    """Load KITTI LiDAR .bin (x,y,z,reflectance) → Nx3 xyz."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # xyz


def make_homog(pts_xyz):
    """Nx3 → 4xN homogeneous."""
    pts_xyz = np.asarray(pts_xyz)
    N = pts_xyz.shape[0]
    out = np.ones((4, N), dtype=np.float64)
    out[:3, :] = pts_xyz.T
    return out


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
    KITTI Raw:
    - calib_cam_to_cam.txt   => P_rect_02, R_rect_00 or R0_rect
    - calib_velo_to_cam.txt  => R, T

    Return:
        P2 (3x4)
        T_cam0_from_velo (4x4)
        T_cam2_from_cam0 (4x4)
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

    T_cam0_from_velo = R_rect_00_4 @ Tr_velo_to_cam

    # Cam0 → Cam2 baseline from P2
    fx = P2[0, 0]
    Tx = P2[0, 3]
    baseline = -Tx / fx

    T_cam2_from_cam0 = np.eye(4, dtype=np.float64)
    T_cam2_from_cam0[0, 3] = baseline

    return P2, T_cam0_from_velo, T_cam2_from_cam0


def project_points(P, X_cam4):
    """Project 4xN cam-space points using 3x4 P."""
    uvw = P @ X_cam4  # (3, N)
    z = uvw[2, :]
    mask = z > 1e-6
    u = uvw[0, mask] / z[mask]
    v = uvw[1, mask] / z[mask]
    return u, v, z[mask], mask


def forward_splat_depth(u, v, z, W, H):
    """
    Simple z-buffer depth splat: per pixel, keep nearest z.
    u, v: float pixel coords, z: depth (meters).
    """
    depth = np.full((H, W), np.inf, dtype=np.float32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    for i in range(len(ui)):
        x = ui[i]
        y = vi[i]
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        d = z[i]
        if d <= 0:
            continue
        if d < depth[y, x]:
            depth[y, x] = d

    depth[depth == np.inf] = 0.0  # 0 = no depth
    return depth


# ------------------ Main export logic ------------------ #

def main(args):
    seq_dir   = os.path.join(args.kitti_root, "sequences", args.sequence)
    img_dir   = os.path.join(seq_dir, "image_2")
    velo_dir  = os.path.join(seq_dir, "velodyne")

    # Load calib (P2 + extrinsics)
    P2, T_cam0_from_velo, T_cam2_from_cam0 = load_raw_calib(args.raw_calib_dir)

    # Use first frame’s image to get H, W
    sample_img = cv2.imread(
        os.path.join(img_dir, f"{args.frame_start:06d}.png")
    )
    if sample_img is None:
        raise RuntimeError("Could not read sample image. Check paths.")
    H, W = sample_img.shape[:2]

    # Output depth folder (inside your Nerfstudio scene root)
    depth_dir = os.path.join(args.scene_root, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    print(f"Saving depth PNGs to: {depth_dir}")

    for t in range(args.frame_start, args.frame_end):
        img_path = os.path.join(img_dir, f"{t:06d}.png")
        velo_path = os.path.join(velo_dir, f"{t:06d}.bin")

        if not os.path.exists(img_path) or not os.path.exists(velo_path):
            print(f"[WARN] Missing image/velodyne for frame {t:06d}, skipping.")
            continue

        # 1) Load LiDAR points in LiDAR frame
        pts_xyz = load_velodyne_bin(velo_path)
        X_velo  = make_homog(pts_xyz)               # 4xN

        # 2) LiDAR → rectified cam0 → cam2
        X_cam0 = T_cam0_from_velo @ X_velo          # 4xN
        X_cam2 = T_cam2_from_cam0 @ X_cam0          # 4xN

        # 3) Project into cam2 image plane
        u, v, z, _ = project_points(P2, X_cam2)     # u,v: pixel coords, z: meters

        # 4) Splat to z-buffer depth map (meters)
        depth_m = forward_splat_depth(u, v, z, W, H)

        # 5) Convert to millimetres for depth-nerfacto (uint16)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0
        depth_mm = (depth_m * 1000.0).round().astype(np.uint16)

        depth_path = os.path.join(depth_dir, f"{t:06d}.png")
        cv2.imwrite(depth_path, depth_mm)
        print(f"[OK] Saved depth for frame {t:06d} → {depth_path}")

    print("All requested frames processed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Export KITTI LiDAR depth maps for depth-nerfacto")
    ap.add_argument("--kitti_root", default= "/home/sriramg/payalsaha/KITTI/dataset/",
                    help="Root of KITTI odometry dataset, e.g. /home/.../KITTI/dataset")
    ap.add_argument("--sequence", default= "00",
                    help="Sequence id, e.g. 00")
    ap.add_argument("--raw_calib_dir", default="/home/sriramg/payalsaha/KITTI/2011_10_03",
                    help="Raw calib date folder, e.g. /home/.../KITTI/2011_10_03")
    ap.add_argument("--scene_root", default="/home/sriramg/payalsaha/kitti_nerf_scene",
                    help="Nerfstudio scene root, e.g. /home/.../kitti_nerf_scene")
    ap.add_argument("--frame_start", type=int, default=0)
    ap.add_argument("--frame_end",   type=int, default=100,
                    help="Non-inclusive: will export frames [start, end)")
    args = ap.parse_args()
    main(args)