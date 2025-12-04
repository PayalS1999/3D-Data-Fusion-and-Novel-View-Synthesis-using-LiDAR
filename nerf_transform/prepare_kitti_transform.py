#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path
import shutil

# ---------- CONFIG: EDIT THESE ----------
KITTI_ROOT   = "/home/sriramg/payalsaha/KITTI/dataset"
SEQUENCE     = "00"
RAW_CALIB    = "/home/sriramg/payalsaha/KITTI/2011_10_03/calib_cam_to_cam.txt"
POSES_FILE   = "/home/sriramg/payalsaha/KITTI/dataset/poses/00.txt"

# Where to create NeRF-style dataset
OUT_DIR      = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5"

# choose a subset of frames so training is not crazy-slow
FRAME_START  = 0
FRAME_END    = 2500  # non-inclusive
FRAME_STEP   = 5     # use every 2nd frame => ~50 images
# ----------------------------------------


def read_kitti_calib_cam2(cam_calib_path):
    """Return P2 (3x4) and T_cam2_from_cam0 (4x4) as in your earlier code."""
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

    cam = parse_keyvals(
        cam_calib_path, keys_keep={"P_rect_02", "R_rect_00", "R0_rect"}
    )

    P2 = cam["P_rect_02"].reshape(3, 4)

    if "R_rect_00" in cam:
        Rrect = cam["R_rect_00"].reshape(3, 3)
    else:
        Rrect = cam["R0_rect"].reshape(3, 3)

    R_rect_00_4 = np.eye(4)
    R_rect_00_4[:3, :3] = Rrect

    # Cam0 -> Cam2 baseline from P2
    fx = P2[0, 0]
    Tx = P2[0, 3]
    baseline = -Tx / fx

    T_cam2_from_cam0 = np.eye(4)
    T_cam2_from_cam0[0, 3] = baseline

    return P2, T_cam2_from_cam0


def load_poses(poses_file):
    """Return list of 4x4 T_w_from_cam0."""
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            assert len(vals) == 12
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def main():
    seq_dir = Path(KITTI_ROOT) / "sequences" / SEQUENCE
    img_dir = seq_dir / "image_2"

    out_dir = Path(OUT_DIR)
    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # intrinsics + cam2 transform from cam0
    P2, T_cam2_from_cam0 = read_kitti_calib_cam2(RAW_CALIB)
    fx = float(P2[0, 0])
    fy = float(P2[1, 1])
    cx = float(P2[0, 2])
    cy = float(P2[1, 2])

    poses = load_poses(POSES_FILE)

    # read first image to get W,H
    sample_img_path = img_dir / f"{FRAME_START:06d}.png"
    import cv2
    sample_img = cv2.imread(str(sample_img_path))
    H, W = sample_img.shape[:2]

    frames = []
    for i in range(FRAME_START, FRAME_END, FRAME_STEP):
        src = img_dir / f"{i:06d}.png"
        if not src.exists():
            print(f"[WARN] image {src} not found, skipping")
            continue

        # copy or symlink
        dst_name = f"{i:06d}.png"  # keep original numbering
        dst = img_out_dir / dst_name
        if not dst.exists():
            shutil.copy(src, dst)

        # pose: cam2 -> world  =  T_w_cam0 @ inv(T_cam2_from_cam0)
        T_w_from_cam0 = poses[i]
        T_cam2_inv = np.linalg.inv(T_cam2_from_cam0)
        T_w_from_cam2 = T_w_from_cam0 @ T_cam2_inv

        frames.append({
            "file_path": f"images/{dst_name}",
            "transform_matrix": T_w_from_cam2.tolist()
        })

    transforms = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
        "camera_model": "OPENCV",
        "frames": frames
    }

    out_json = out_dir / "transforms.json"
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"[OK] Wrote {out_json} with {len(frames)} frames.")


if __name__ == "__main__":
    main()