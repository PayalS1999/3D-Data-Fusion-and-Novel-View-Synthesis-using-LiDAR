import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import (
    load_velodyne_bin,
    project_points,
)

def _parse_kitti_keyvals(txt_path, keys_keep=None):
    """Parse 'key: v1 v2 ...' lines into dict[str, np.ndarray]."""
    out = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, vals = line.split(":", 1)
            k = k.strip()
            if keys_keep is not None and k not in keys_keep:
                continue
            nums = [float(x) for x in vals.strip().split()]
            out[k] = np.array(nums, dtype=np.float64)
    return out

def load_raw_cam_velo_extrinsics(raw_calib_dir):
    """
    Returns:
      P2            : (3,4) projection for cam2 (left color)
      R_rect_00_4x4 : (4,4) rectification
      Tr_velo_to_cam_4x4 : (4,4) extrinsic Velodyne->Cam0
      T_cam2_from_velo   : (4,4) total Velodyne->Cam2 (rectified)
    Expects:
      raw_calib_dir/calib_cam_to_cam.txt
      raw_calib_dir/calib_velo_to_cam.txt
    Optionally:
      raw_calib_dir/calib_imu_to_velo.txt (not required here)
    """

    f_cam  = os.path.join(raw_calib_dir, "calib_cam_to_cam.txt")
    f_velo = os.path.join(raw_calib_dir, "calib_velo_to_cam.txt")

    # --- cam_to_cam: intrinsics + rectification
    cam = _parse_kitti_keyvals(
        f_cam,
        keys_keep={"P_rect_00","P_rect_01","P_rect_02","P_rect_03","R_rect_00","R0_rect"}  # some files use R0_rect
    )
    # Prefer P_rect_02 for image_2
    if "P_rect_02" not in cam:
        raise ValueError("P_rect_02 not found in calib_cam_to_cam.txt (needed for camera 2).")
    P2 = cam["P_rect_02"].reshape(3, 4)

    if "R_rect_00" in cam:
        R_rect = cam["R_rect_00"].reshape(3, 3)
    elif "R0_rect" in cam:
        R_rect = cam["R0_rect"].reshape(3, 3)
    else:
        raise ValueError("R_rect_00 / R0_rect not found in calib_cam_to_cam.txt.")

    R_rect_00_4x4 = np.eye(4, dtype=np.float64); R_rect_00_4x4[:3,:3] = R_rect

    # --- velo_to_cam: extrinsics
    velo = _parse_kitti_keyvals(f_velo, keys_keep={"R","T"})
    if "R" not in velo or "T" not in velo:
        raise ValueError("R/T not found in calib_velo_to_cam.txt.")
    R = velo["R"].reshape(3, 3)
    T = velo["T"].reshape(3,)

    Tr_velo_to_cam_4x4 = np.eye(4, dtype=np.float64)
    Tr_velo_to_cam_4x4[:3,:3] = R
    Tr_velo_to_cam_4x4[:3, 3] = T

    # Total Velodyne -> rectified Cam0, then same rectification applies to all cams
    T_cam_from_velo = R_rect_00_4x4 @ Tr_velo_to_cam_4x4

    return P2, R_rect_00_4x4, Tr_velo_to_cam_4x4, T_cam_from_velo


def read_calib(path):
    """
       Reads KITTI calib.txt into a dict of name -> numpy array.
       Handles P0..P3
       """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            key, value = line.split(":", 1)
            val = value.strip()
            nums = [float(x) for x in val.split()]

            if key.startswith('P'):
                data[key] = np.array(nums, dtype=np.float64).reshape(3,4)
    return data


def make_homogenous(ptz_xyz ):
    """
    (N,3) -> (4,N) homogeneous
    """

    N = ptz_xyz.shape[0]
    pts = np.ones((4, N), dtype=np.float64)
    pts[:3,:] = ptz_xyz.T
    return pts


def z_buffer_depth(u,v,z,imgW, imgH):
    """
        Build a z-buffer (depth image) from projected points.
        For each pixel, keep the smallest depth (closest surface).
        """
    depth = np.full((imgH,imgW), np.inf, dtype=np.float32)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    in_bounds = (ui >= 0) & (ui < imgW) & (vi > 0) & (vi < imgH)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]

    idx = vi * imgW + ui
    for j in range(len(idx)):
        yx = (vi[j], ui[j])
        if z[j] < depth[yx]:
            depth[yx] = z[j]

    depth[~np.isfinite(depth)] = 0.0
    return depth

def colorize_depth(depth):
    """
    Simple depth colorization for visualization (near = bright)
    """
    d = depth.copy()
    d[d <= 0] = np.nan
    vmax = np.nanpercentile(d, 95) if np.isnan(d).sum() < d.size else 1.0
    norm = np.clip(d/vmax, 0, 1)
    norm = np.nan_to_num(norm, nan = 0.0)
    vis = (1.0 - norm) * 255
    return vis.astype(np.uint8)

def main(args):
    seq_dir = os.path.join(args.kitti_root, "sequences", args.sequence)
    img_path = os.path.join(seq_dir, "image_2", f"{args.frame}.png")
    velo_path = os.path.join(seq_dir, "velodyne", f"{args.frame}.bin")
    calib_path = os.path.join(seq_dir, "calib.txt")

    outdir = 'Depth_plots'
    os.makedirs(outdir, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(img.shape)
    imgH, imgW = img.shape[:2]
    pts = load_velodyne_bin(velo_path)[: , :3]

    calib = read_calib(calib_path)

    P2, R_rect_00_4x4, Tr_velo_to_cam_4x4, T_cam_from_velo = load_raw_cam_velo_extrinsics(args.raw_calib_dir)
    pts_h = make_homogenous(pts)  # (4,N)
    pts_cam = T_cam_from_velo @ pts_h

    in_front = pts_cam[2, :] > 1e-6
    pts_cam = pts_cam[:, in_front]

    u, v, z, mask = project_points(P2, pts_cam)
    overlay = img.copy()

    for (uu, vv) in zip(u, v):
        ui, vi = int(round(uu)), int(round(vv))
        if 0 <= ui < imgW and 0 <= vi < imgH:
            cv2.circle(overlay, (ui, vi), 1, (0, 255, 0), -1)

    out_overlay = f"{outdir}/overlay_points_{args.frame}.png"
    cv2.imwrite(out_overlay, overlay)
    print(f"[OK] Saved projection overlay -> {out_overlay}")

    # Build and save a depth image from projected points
    depth = z_buffer_depth(u, v, z, imgW, imgH)
    depth_vis = colorize_depth(depth)
    # put side-by-side with the RGB for quick inspection
    rgb_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Projected LiDAR on Image")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Depth Image (z-buffer from LiDAR)")
    plt.imshow(depth_vis, cmap="gray")
    plt.axis("off")
    out_depth = f"{outdir}/depth_image_{args.frame}.png"
    plt.tight_layout()
    plt.savefig(out_depth, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved depth visualization -> {out_depth}")

    # ----------------------------
    # Optional: project into a NEW viewpoint (if poses available)
    # ----------------------------
    if args.poses_file and args.target_frame is not None:
        # Assumes poses file gives T_world_from_cam (for the SAME camera as P2).
        # Many KITTI odometry pose files are for the grayscale camera (cam0).
        # If your poses are for cam0 while your P2 is cam2, you need a fixed extrinsic T_cam0_to_cam2.
        # For simplicity here, we assume the poses match P2's camera. Adjust if needed.
        poses = []
        with open(args.poses_file, "r") as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) != 12:
                    raise ValueError("Pose line must have 12 floats (3x4).")
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                poses.append(T)
        src_idx = int(args.frame)
        tgt_idx = int(args.target_frame)
        if src_idx >= len(poses) or tgt_idx >= len(poses):
            raise IndexError("Pose indices out of range for provided poses file.")

        T_w_from_cam_src = poses[src_idx]
        T_w_from_cam_tgt = poses[tgt_idx]

        # We have points in LiDAR. Convert LiDAR -> source cam -> world -> target cam.
        T_cam_src_from_velo = T_cam_from_velo
        T_w_from_velo = T_w_from_cam_src @ np.linalg.inv(T_cam_src_from_velo)
        T_cam_tgt_from_w = np.linalg.inv(T_w_from_cam_tgt)

        T_cam_tgt_from_velo = T_cam_tgt_from_w @ T_w_from_velo

        pts_cam_tgt = T_cam_tgt_from_velo @ pts_h

        u2, v2, z2, mask2 = project_points(P2, pts_cam_tgt)

        overlay2 = img.copy()
        for (uu, vv) in zip(u2, v2):
            ui, vi = int(round(uu)), int(round(vv))
            if 0 <= ui < imgW and 0 <= vi < imgH:
                cv2.circle(overlay2, (ui, vi), 1, (255, 0, 0), -1)
        out_overlay2 = f"{outdir}/overlay_points_{args.frame}_to_{args.target_frame}.png"
        cv2.imwrite(out_overlay2, overlay2)
        print(f"[OK] Saved NEW-viewpoint overlay -> {out_overlay2}")

        depth2 = z_buffer_depth(u2, v2, z2, imgW, imgH)
        depth_vis2 = colorize_depth(depth2)
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Projected LiDAR on Image (to frame {args.target_frame})")
        plt.imshow(cv2.cvtColor(overlay2, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Depth Image (new viewpoint)")
        plt.imshow(depth_vis2, cmap="gray")
        plt.axis("off")
        out_depth2 = f"{outdir}/depth_image_{args.frame}_to_{args.target_frame}.png"
        plt.tight_layout()
        plt.savefig(out_depth2, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved NEW-viewpoint depth visualization -> {out_depth2}")
    else:
        print("[Info] Skipping new-viewpoint projection (no poses file / target frame provided).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project KITTI LiDAR into camera image plane and visualize.")
    parser.add_argument("--kitti_root", default= "/home/sriramg/payalsaha/KITTI/dataset/", help="Root of KITTI odometry dataset (contains 'sequences').")
    parser.add_argument("--sequence", default= "00", help="Sequence id, e.g., 00")
    parser.add_argument("--frame", default = "000000", help="Frame id, e.g., 000000")
    parser.add_argument("--poses_file", default="/home/sriramg/payalsaha/KITTI/dataset/poses/00.txt", help="Optional: path to poses/XX.txt (if available).")
    parser.add_argument("--target_frame", default="000007", help="Optional: different frame index for new viewpoint.")
    parser.add_argument("--raw_calib_dir", default="/home/sriramg/payalsaha/KITTI/2011_10_03", help="Optional: different frame index for new viewpoint.")
    args = parser.parse_args()
    main(args)











