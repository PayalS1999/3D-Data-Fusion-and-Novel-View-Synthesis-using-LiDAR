#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import time

###############################################################################
# ------------------------------ BASIC UTILITIES ------------------------------
###############################################################################

def load_velodyne_bin(bin_path):
    """Load xyz or xyzr KITTI LiDAR."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # xyz only


def make_homog(pts_xyz):
    """Convert Nx3 -> 4xN homogeneous."""
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    N = pts_xyz.shape[0]
    out = np.ones((4, N), dtype=np.float32)
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
            out[k] = np.array(nums, dtype=np.float32)

    return out


def load_raw_calib(raw_dir):
    """
    KITTI Raw:
    - calib_cam_to_cam.txt   => P_rect_02, R_rect_00
    - calib_velo_to_cam.txt  => R, T
    Return:
        P2 (3x4)
        T_cam0_from_velo (4x4)
        T_cam2_from_cam0 (4x4)
    """
    f_cam = os.path.join(raw_dir, "calib_cam_to_cam.txt")
    f_velo = os.path.join(raw_dir, "calib_velo_to_cam.txt")

    cam = parse_keyvals(
        f_cam, keys_keep={"P_rect_02", "R_rect_00", "R0_rect"}
    )

    P2 = cam["P_rect_02"].reshape(3, 4)

    # rectification
    if "R_rect_00" in cam:
        Rrect = cam["R_rect_00"].reshape(3, 3)
    else:
        Rrect = cam["R0_rect"].reshape(3, 3)

    R_rect_00_4 = np.eye(4)
    R_rect_00_4[:3, :3] = Rrect

    # velo->cam0
    velo = parse_keyvals(f_velo, keys_keep={"R", "T"})
    R = velo["R"].reshape(3, 3)
    T = velo["T"].reshape(3)

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3] = T

    T_cam0_from_velo = R_rect_00_4 @ Tr_velo_to_cam

    # Cam0 -> Cam2 translation from P_rect_02
    fx = P2[0, 0]
    Tx = P2[0, 3]
    baseline = -Tx / fx

    T_cam2_from_cam0 = np.eye(4)
    T_cam2_from_cam0[0, 3] = baseline

    return P2, T_cam0_from_velo, T_cam2_from_cam0


def project_points(P, X_cam4):
    uvw = P @ X_cam4  # (3, N)
    z = uvw[2, :]
    mask = z > 1e-6
    u = uvw[0, mask] / z[mask]
    v = uvw[1, mask] / z[mask]
    return u, v, z[mask], mask


def bilinear_sample(img, u, v):
    """Returns Nx3 BGR colors."""
    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    H, W = img.shape[:2]
    N = len(u)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # clip
    u0c = np.clip(u0, 0, W - 1)
    u1c = np.clip(u1, 0, W - 1)
    v0c = np.clip(v0, 0, H - 1)
    v1c = np.clip(v1, 0, H - 1)

    # weights
    du = (u - u0).reshape(N, 1)
    dv = (v - v0).reshape(N, 1)

    c00 = img[v0c, u0c, :].astype(np.float32)
    c10 = img[v0c, u1c, :].astype(np.float32)
    c01 = img[v1c, u0c, :].astype(np.float32)
    c11 = img[v1c, u1c, :].astype(np.float32)

    c0 = c00 * (1 - du) + c10 * du
    c1 = c01 * (1 - du) + c11 * du
    c = c0 * (1 - dv) + c1 * dv
    return np.clip(c, 0, 255).astype(np.uint8)


###############################################################################
# -------------------------- MULTI-FRAME ACCUMULATION --------------------------
###############################################################################

def accumulate_frames(frames, seq_dir, poses, T_cam0_from_velo, downsample_ratio=1.0):
    """
    Return:
        all_world   : 4xN homogeneous world coords
        all_frame_ids : (N,) frame index per point
        all_intensity : (N,) reflectance per point in [0,1]
    """
    all_world = []
    all_frame_ids = []
    all_intensity = []

    for k in frames:
        velo_path = os.path.join(seq_dir, "velodyne", f"{k:06d}.bin")
        raw = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        pts_xyz = raw[:, :3]
        reflec = raw[:, 3]

        # 🔹 Random downsampling for sparsity ablation
        if downsample_ratio < 1.0:
            mask = np.random.rand(pts_xyz.shape[0]) < downsample_ratio
            pts_xyz = pts_xyz[mask]
            reflec = reflec[mask]

        X_velo = make_homog(pts_xyz)

        T_w_from_cam0_k = poses[k]
        T_w_from_velo_k = T_w_from_cam0_k @ np.linalg.inv(T_cam0_from_velo)
        X_world = T_w_from_velo_k @ X_velo  # (4,N)

        all_world.append(X_world)
        all_frame_ids.append(np.full(X_world.shape[1], k, dtype=np.int32))
        all_intensity.append(reflec)

    all_world = np.hstack(all_world)
    all_frame_ids = np.concatenate(all_frame_ids)
    all_intensity = np.concatenate(all_intensity)

    # Normalize intensity to [0,1] (avoid divide-by-zero)
    if all_intensity.size > 0:
        imin, imax = float(all_intensity.min()), float(all_intensity.max())
        if imax > imin:
            all_intensity = (all_intensity - imin) / (imax - imin)
        else:
            all_intensity = np.ones_like(all_intensity, dtype=np.float32)
    else:
        all_intensity = np.zeros_like(all_frame_ids, dtype=np.float32)

    return all_world, all_frame_ids, all_intensity


def temporal_weights(frame_ids, target_frame):
    """w = exp(-|k - t|)."""
    return np.exp(-np.abs(frame_ids - target_frame))


###############################################################################
# ----------------------------- FORWARD SPLATTING ------------------------------
###############################################################################

def forward_splat_weighted(u, v, z, colors, weights, W, H):
    depth = np.full((H, W), np.inf, dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.float32)
    hit = np.zeros((H, W), dtype=np.uint8)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    for i in range(len(ui)):
        x = ui[i]; y = vi[i]
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        d = z[i]
        if d < depth[y, x]:
            depth[y, x] = d
            color[y, x] = colors[i] * weights[i]
            hit[y, x] = 255

    depth[depth == np.inf] = 0
    return color.astype(np.uint8), depth, hit

def forward_splat_surfel(u, v, z, colors, weights, W, H, radius_px=2):
    """
    Surfel splatting: each point becomes a small disc of radius_px pixels.
    Still uses z-buffer: nearest disc wins.
    """
    depth = np.full((H, W), np.inf, dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.float32)
    hit   = np.zeros((H, W), dtype=np.uint8)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    r = int(radius_px)
    r2 = r * r

    for i in range(len(ui)):
        cx, cy, d, w = ui[i], vi[i], z[i], weights[i]
        if d <= 0:
            continue
        if cx < -r or cx >= W + r or cy < -r or cy >= H + r:
            continue

        x0 = max(0, cx - r)
        x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(H - 1, cy + r)

        base_color = colors[i].astype(np.float32) * float(w)

        for yy in range(y0, y1 + 1):
            dy = yy - cy
            dy2 = dy * dy
            for xx in range(x0, x1 + 1):
                dx = xx - cx
                if dx * dx + dy2 > r2:
                    continue  # outside disc
                if d < depth[yy, xx]:
                    depth[yy, xx] = d
                    color[yy, xx] = base_color
                    hit[yy, xx]   = 255

    depth[depth == np.inf] = 0.0
    return color.astype(np.uint8), depth, hit


def forward_splat_gaussian(u, v, z, colors, weights, W, H,
                           radius_px=3, depth_margin=0.5):
    """
    Gaussian surfel splatting:
    - Each point is a 2D Gaussian in screen space.
    - We maintain per-pixel accum_color/accum_weight.
    - Simple depth handling: if a new Gaussian is significantly closer
      (by depth_margin meters), it replaces the previous layer.
      If similar depth, we blend.
    """
    # Make sure everything is 1D and aligned
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    z = np.asarray(z).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    H = int(H)
    W = int(W)

    accum_color  = np.zeros((H, W, 3), dtype=np.float32)
    accum_weight = np.zeros((H, W), dtype=np.float32)
    depth        = np.full((H, W), np.inf, dtype=np.float32)
    hit          = np.zeros((H, W), dtype=np.uint8)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    r = int(radius_px)
    two_sigma2 = 2.0 * (r * 0.5) ** 2  # sigma ~ r/2

    for i in range(len(ui)):
        cx = ui[i]
        cy = vi[i]
        d  = z[i]
        w  = float(weights[i])  # force scalar

        if d <= 0:
            continue
        if cx < -r or cx >= W + r or cy < -r or cy >= H + r:
            continue

        x0 = max(0, cx - r)
        x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(H - 1, cy + r)

        base_col = colors[i].astype(np.float32)

        for yy in range(y0, y1 + 1):
            dy = yy - cy
            dy2 = dy * dy
            for xx in range(x0, x1 + 1):
                dx = xx - cx
                r2 = dx * dx + dy2
                if r2 > r * r:
                    continue

                # 2D Gaussian weight in screen space
                g = np.exp(-r2 / two_sigma2) * w  # scalar
                #if g < 1e-4:
                #    continue

                # simple depth logic: if much closer, reset; if close in depth, blend
                if d + depth_margin < depth[yy, xx]:
                    depth[yy, xx]        = d
                    accum_color[yy, xx]  = base_col * g
                    accum_weight[yy, xx] = g
                    hit[yy, xx]          = 255
                elif abs(d - depth[yy, xx]) <= depth_margin:
                    accum_color[yy, xx]  += base_col * g
                    accum_weight[yy, xx] += g
                    hit[yy, xx]           = 255
                # if much farther, ignore (occluded)

    mask = accum_weight > 0
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[mask] = (accum_color[mask] / accum_weight[mask, None]).clip(0, 255).astype(np.uint8)

    depth[~mask] = 0.0
    return out, depth, hit

def compute_density_confidence(X_world, voxel_size=0.5):
    """
    X_world: 4xN homogeneous world coords
    voxel_size: meters; controls neighborhood size
    Returns: (N,) confidence in [0,1] based on voxel population.
    """
    xyz = X_world[:3, :].T  # (N,3)
    if xyz.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # Quantize to voxel grid
    vox = np.floor(xyz / float(voxel_size)).astype(np.int32)

    # Use unique voxels + counts
    # Convert to structured array so np.unique works row-wise
    vox_view = np.ascontiguousarray(vox).view(
        np.dtype((np.void, vox.dtype.itemsize * vox.shape[1]))
    )
    uniq, inv, counts = np.unique(vox_view, return_inverse=True, return_counts=True)

    # For each point, find count of its voxel
    local_count = counts[inv].astype(np.float32)

    # Map counts -> [0,1] (more neighbors = higher confidence)
    c_min, c_max = float(local_count.min()), float(local_count.max())
    if c_max > c_min:
        conf = (local_count - c_min) / (c_max - c_min)
    else:
        conf = np.ones_like(local_count, dtype=np.float32)

    return conf


def hole_fill(color_img, mask, iterations=3):
    kernel = np.ones((3, 3), np.uint8)
    filled = color_img.copy()
    inv = cv2.bitwise_not(mask)

    for _ in range(iterations):
        dil = cv2.dilate(filled, kernel, iterations=1)
        for c in range(3):
            ch = filled[:, :, c]
            ch[inv > 0] = dil[:, :, c][inv > 0]
            filled[:, :, c] = ch
        inv = cv2.erode(inv, kernel, iterations=1)

    return filled

Gaussian_params = {
    "gauss_radius": 3,
    "gauss_depth_margin": 0.5,
}

Surfel_params = {
    "surfel_radius": 2
}

###############################################################################
# ---------------------------------- MAIN --------------------------------------
###############################################################################

def main(args):
    t0 = time.time()
    seq_dir = os.path.join(args.kitti_root, "sequences", args.sequence)
    img_dir = os.path.join(seq_dir, "image_2")

    # Load calib
    P2, T_cam0_from_velo, T_cam2_from_cam0 = load_raw_calib(args.raw_calib_dir)

    # Load poses
    poses = []
    with open(args.poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)

    t = int(args.target_frame)
    # Use temporal window controlled by --num_neighbors
    frames = list(range(t - args.num_neighbors, t + args.num_neighbors + 1))

    # Load images
    img_tgt = cv2.imread(os.path.join(img_dir, f"{t:06d}.png"))
    H, W = img_tgt.shape[:2]

    # 1) accumulate frames in world coords
    world_points, frame_ids, intensities = accumulate_frames(
        frames, seq_dir, poses, T_cam0_from_velo,
        downsample_ratio=args.downsample_ratio
    )

    # 2) project world -> target cam2
    T_w_from_cam0_t = poses[t]
    X_cam2_t = T_cam2_from_cam0 @ np.linalg.inv(T_w_from_cam0_t) @ world_points
    u, v, z, mask = project_points(P2, X_cam2_t)

    # 3) keep only points that are actually visible in the target view
    world_points_vis = world_points[:, mask]
    frame_ids_vis = frame_ids[mask]
    intensities_vis = intensities[mask]

    # 4) temporal weights only for visible points
    weights_temporal = temporal_weights(frame_ids_vis, t)

    # 5) depth-confidence **only on visible points** (MUCH smaller set)
    if args.use_confidence:
        print("Computing density confidence on", world_points_vis.shape[1], "points...")
        conf_density = compute_density_confidence(world_points_vis, voxel_size=args.voxel_size)
        print(f"[density] {world_points_vis.shape[1]} pts, time = {time.time() - t0:.3f}s")

        # intensity in [0,1] already
        conf_intensity = intensities_vis.astype(np.float32)

        weights_temporal = np.asarray(weights_temporal, dtype=np.float32).ravel()
        conf_density = np.asarray(conf_density, dtype=np.float32).ravel()
        conf_intensity = np.asarray(conf_intensity, dtype=np.float32).ravel()

        conf_total = conf_density * (0.5 + 0.5 * conf_intensity)
    else:
        conf_total = np.ones_like(weights_temporal, dtype=np.float32)

    weights = weights_temporal * conf_total  # final per-point weights

    # 6) from here on, use u, v, z, weights, frame_ids_vis, etc.
    frame_ids = frame_ids_vis
    intensities = intensities_vis

    # ----------------------
    # 4. Sample colors from corresponding source frames
    # ----------------------
    colors = []
    for k in frames:
        # mask: which points came from frame k
        mk = (frame_ids == k)
        if mk.sum() == 0:
            continue

        img_src = cv2.imread(os.path.join(img_dir, f"{k:06d}.png"))
        uk = u[mk]; vk = v[mk]

        ck = bilinear_sample(img_src, uk, vk)
        colors.append((mk, ck))

    # assemble final colors
    colors_final = np.zeros((u.shape[0], 3), dtype=np.uint8)
    for mk, ck in colors:
        colors_final[mk] = ck

    # ----------------------
    # 5. Forward splat with z-test + temporal weights
    # ----------------------
    if args.splat_type == "point":
        color_splat, depth_splat, hit = forward_splat_weighted(
            u, v, z, colors_final, weights, W, H
        )
    elif args.splat_type == "surfel":
        color_splat, depth_splat, hit = forward_splat_surfel(
            u, v, z, colors_final, weights, W, H,
            radius_px= Surfel_params["surfel_radius"],
        )
    else:  # gaussian
        color_splat, depth_splat, hit = forward_splat_gaussian(
            u, v, z, colors_final, weights, W, H,
            radius_px= Gaussian_params["gauss_radius"], #args.gauss_radius,
            depth_margin=Gaussian_params["gauss_depth_margin"],
        )

    # ----------------------
    # 6. Hole filling
    # ----------------------
    color_filled = hole_fill(color_splat, hit, iterations=3)

    # ----------------------
    # 7. Save outputs
    # ----------------------
    if args.downsample_ratio == 1.0:
        tail = "d1"
    elif args.downsample_ratio == 0.1:
        tail  = "d0p1"
    else:
        tail = "d0p5"
    outdir = f"depth_conf_ablation_studies_{args.splat_type}_{2 ** args.num_neighbors - 1}frames_{tail}"
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(f"{outdir}/multi_splat_color_{t:06d}.png", color_splat)
    cv2.imwrite(f"{outdir}/multi_splat_filled_{t:06d}.png", color_filled)
    cv2.imwrite(f"{outdir}/multi_splat_depth_{t:06d}.png", (depth_splat / depth_splat.max() * 255).astype(np.uint8))

    print(f"Settings: splat type = {args.splat_type}, num neighbors = {args.num_neighbors}, downsample ratio = {args.downsample_ratio}")
    print("[OK] Saved multi-frame splat images.")

    # ----------------------
    # 8. Compute PSNR / SSIM (target frame must exist)
    # ----------------------
    psnr = peak_signal_noise_ratio(img_tgt, color_filled)
    ssim = structural_similarity(img_tgt, color_filled, channel_axis=2)

    print("PSNR:", psnr)
    print("SSIM:", ssim)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Multi-frame KITTI NVS rendering")
    ap.add_argument("--kitti_root", default= "/home/sriramg/payalsaha/KITTI/dataset/")
    ap.add_argument("--sequence", default= "00")
    ap.add_argument("--target_frame", default="000007")
    ap.add_argument("--raw_calib_dir",  default="/home/sriramg/payalsaha/KITTI/2011_10_03")
    ap.add_argument("--poses_file", default="/home/sriramg/payalsaha/KITTI/dataset/poses/00.txt")
    ap.add_argument("--splat_type", default="gaussian", choices=["point", "surfel", "gaussian"],
                    metavar= "gaussian/point/surfel")
    ap.add_argument(
        "--use_confidence",
        action="store_true",
        help="Enable depth-confidence weighting (voxel density + optional intensity).",
    )
    ap.add_argument(
        "--voxel_size",
        type=float,
        default=0.5,
        help="Voxel size in meters for density-based confidence.",
    )
    ap.add_argument(
        "--downsample_ratio",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of LiDAR points (1.0 = no downsampling).",
    )
    ap.add_argument(
        "--num_neighbors",
        type=int,
        default=1,
        help="Temporal neighbors on each side: 0=single frame, 1=3 frames, 2=5 frames, etc.",
    )
    args = ap.parse_args()
    main(args)