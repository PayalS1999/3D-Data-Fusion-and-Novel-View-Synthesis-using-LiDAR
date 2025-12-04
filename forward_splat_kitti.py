from utils import (
    load_velodyne_bin,
    project_points,
    compute_psnr,
    compute_ssim
)

import argparse, os
import numpy as np
import cv2

# -----------------------
# I/O helpers
# -----------------------
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(path)
    return img

def parse_keyvals(txt_path, keys_keep=None):
    out = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line: continue
            k, vals = line.split(":", 1)
            k = k.strip()
            if keys_keep and k not in keys_keep: continue
            nums = [float(x) for x in vals.strip().split()]
            out[k] = np.array(nums, dtype=np.float64)
    return out

# -----------------------
# Calibration
# -----------------------

def make_homog(xyz):
    # Accept (N,3) or (N,4) [x,y,z,(r)]
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected (N,3/4), got {xyz.shape}")
    xyz3 = xyz[:, :3]
    N = xyz3.shape[0]
    h = np.ones((4, N), dtype=np.float64)
    h[:3, :] = xyz3.T
    return h

def load_raw_calib(raw_dir):
    """Return P2 (3x4), R_rect_00_4 (4x4), Tr_velo_to_cam_4 (4x4),
       and T_cam2_from_cam0 (4x4) for rectified cameras."""
    f_cam  = os.path.join(raw_dir, "calib_cam_to_cam.txt")
    f_velo = os.path.join(raw_dir, "calib_velo_to_cam.txt")

    cam = parse_keyvals(f_cam, keys_keep={"P_rect_00","P_rect_01","P_rect_02","P_rect_03","R_rect_00","R0_rect"})
    if "P_rect_02" not in cam:
        raise ValueError("P_rect_02 missing in calib_cam_to_cam.txt")
    P2 = cam["P_rect_02"].reshape(3,4)

    Rrect = (cam["R_rect_00"] if "R_rect_00" in cam else cam["R0_rect"]).reshape(3,3)
    R_rect_00_4 = np.eye(4); R_rect_00_4[:3,:3] = Rrect

    velo = parse_keyvals(f_velo, keys_keep={"R","T"})
    R = velo["R"].reshape(3,3)
    T = velo["T"].reshape(3)
    Tr_velo_to_cam_4 = np.eye(4); Tr_velo_to_cam_4[:3,:3] = R; Tr_velo_to_cam_4[:3,3] = T

    # Cam0->Cam2 in rectified space: rotation ~ I, translation along x = baseline from P2
    fx = P2[0,0]; Tx = P2[0,3]  # note: P = K [I|t], where t = [Tx,0,0]^T and baseline B = -Tx/fx
    B = -Tx / fx
    T_cam2_from_cam0 = np.eye(4)
    T_cam2_from_cam0[0,3] = B

    return P2, R_rect_00_4, Tr_velo_to_cam_4, T_cam2_from_cam0

def bilinear_sample(img, u, v):
    """img BGR (H,W,3), u,v float arrays in pixel coords. Returns colors Nx3 (BGR) in [0,255]."""
    H,W = img.shape[:2]
    u0 = np.floor(u).astype(np.int32).ravel()
    v0 = np.floor(v).astype(np.int32).ravel()
    u1 = u0 + 1
    v1 = v0 + 1

    N = u.shape[0]

    u0c, u1c = np.clip(u0, 0, W-1), np.clip(u1, 0, W-1)
    v0c, v1c = np.clip(v0, 0, H-1), np.clip(v1, 0, H-1)

    du = (u-u0).reshape(N,1)
    dv = (v-v0).reshape(N,1)

    c00 = img[v0c, u0c, :].astype(np.float32)
    c10 = img[v0c, u1c, :].astype(np.float32)
    c01 = img[v1c, u0c, :].astype(np.float32)
    c11 = img[v1c, u1c, :].astype(np.float32)

    c0 = c00 * (1-du) + c10 * du
    c1 = c01 * (1-du) + c11 * du
    c = c0 * (1-dv) + c1 * dv

    return np.clip(c, 0, 255).astype(np.uint8)

def forward_splat(u, v, z, colors_bgr, W, H, radius=1):
    """Simple z-buffer splatting.
    Inputs:
        u, v: 1D arrays of pixel coordinates (float)
        z   : 1D array of depths (same length)
        colors_bgr: (N,3) uint8 or float32
    Returns:
        color_img (H,W,3), depth_img (H,W), hit_mask (H,W)
    """
    depth = np.full((H, W), np.inf, dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.float32)
    hit   = np.zeros((H, W), dtype=np.uint8)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    for i in range(len(ui)):
        x = ui[i]
        y = vi[i]
        d = z[i]

        # skip invalid / behind camera
        if d <= 0:
            continue
        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        # small square splat around (x,y)
        x0 = max(0, x - radius)
        x1 = min(W - 1, x + radius)
        y0 = max(0, y - radius)
        y1 = min(H - 1, y + radius)

        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if d < depth[yy, xx]:
                    depth[yy, xx] = d
                    color[yy, xx, :] = colors_bgr[i]
                    hit[yy, xx] = 255

    depth[np.isinf(depth)] = 0.0
    return color.astype(np.uint8), depth, hit

def normalize_depth_for_vis(depth):
    d = depth.copy()
    d[d<=0] = np.nan
    if np.isnan(d).sum() == d.size:
        return np.zeros_like(depth, dtype = np.uint8)
    vmax = np.nanpercentile(d, 95)
    vmax = max(vmax, 1e-3)
    vis = (1.0 - np.clip(d/vmax, 0, 1)) * 255
    return vis.astype(np.uint8)

def hole_fill(color_img, mask, iterations=3):
    """Simple hole filling via morphological dilation of colors guided by mask."""
    filled = color_img.copy()
    kernel = np.ones((3,3),np.uint8)
    inv = cv2.bitwise_not(mask)
    for _ in range(iterations):
        dil = cv2.dilate(filled, kernel, iterations = 1)
        for c in range(3):
            ch = filled[:,:,c]
            ch[inv>0] = dil[:,:,c][inv>0]
            filled[:,:,c] = ch
        inv = cv2.erode(inv, kernel, iterations = 1)
    return filled

def main(args):
    seq_dir = os.path.join(args.kitti_root, "sequences", args.sequence)
    img_src_path = os.path.join(seq_dir, "image_2", f"{args.src_frame}.png")
    img_tgt_path = os.path.join(seq_dir, "image_2", f"{args.tgt_frame}.png")
    velo_src_path = os.path.join(seq_dir, "velodyne", f"{args.src_frame}.bin")

    # Load data
    img_src = load_img(img_src_path)
    img_tgt = load_img(img_tgt_path)
    H, W = img_tgt.shape[:2]
    pts_src = load_velodyne_bin(velo_src_path)

    # Calib
    P2, R_rect_00_4, Tr_velo_to_cam_4, T_cam2_from_cam0 = load_raw_calib(args.raw_calib_dir)
    T_cam0_from_velo = R_rect_00_4 @ Tr_velo_to_cam_4

    # Poses (cam0)
    poses = []
    with open(args.poses_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            T = np.eye(4); T[:3,:4] = np.array(vals, dtype=np.float64).reshape(3,4)
            poses.append(T)
    k = int(args.src_frame); t = int(args.tgt_frame)
    T_w_from_cam0_k = poses[k]
    T_w_from_cam0_t = poses[t]

    # ---- Per-point color from SOURCE image (cam2 at frame k) ----
    # LiDAR(k) -> cam0(k) -> cam2(k) -> project to source image -> sample colors
    P2_src = P2  # same intrinsics across frames in rectified setup
    X_velo_k = make_homog(pts_src)  # (4, N)

    # Match multi-frame logic: use inverse of T_cam0_from_velo
    X_cam0_k = np.linalg.inv(T_cam0_from_velo) @ X_velo_k
    X_cam2_k = T_cam2_from_cam0 @ X_cam0_k
    u_s, v_s, z_s, m_s = project_points(P2_src, X_cam2_k)
    in_bounds_s = (u_s>=0)&(u_s<W)&(v_s>=0)&(v_s<H)&(z_s>0)
    u_s = u_s[in_bounds_s]; v_s = v_s[in_bounds_s]
    X_cam0_k = X_cam0_k[:, m_s][:, in_bounds_s]  # keep aligned subset
    colors = bilinear_sample(img_src, u_s, v_s)   # (Ns,3) BGR

    # ---- Move those colored points to TARGET cam2(t) ----
    # world <- cam0(k)
    T_w_from_velo_k = T_w_from_cam0_k @ np.linalg.inv(T_cam0_from_velo)
    X_world = T_w_from_velo_k @ X_velo_k  # use same X_velo_k
    X_world_sub = X_world[:, m_s][:, in_bounds_s]
    # cam2(t) <- world
    T_cam2t_from_w = T_cam2_from_cam0 @ np.linalg.inv(T_w_from_cam0_t)
    # cam2(t) <- points(world)  where points(world) come from LiDAR(k)
    # X_world = T_w_from_velo_k @ make_homog(pts_src)  # (4,N_all)
    # keep same subset as colored ones (indices must match)
    #X_world_sub = X_world[:, m_s][:, in_bounds_s]
    X_cam2_t = T_cam2t_from_w @ X_world_sub

    # ---- Project into TARGET image & forward splat ----
    u_t, v_t, z_t, m_t = project_points(P2, X_cam2_t)
    u_t = u_t; v_t = v_t; z_t = z_t
    colors = colors[m_t]
    # bounds
    in_bounds_t = (u_t>=0)&(u_t<W)&(v_t>=0)&(v_t<H)&(z_t>0)
    u_t = u_t[in_bounds_t]; v_t = v_t[in_bounds_t]; z_t = z_t[in_bounds_t]
    colors_t = colors[in_bounds_t]

    print("num source LiDAR points:", pts_src.shape[0])
    print("after source proj (u_s):", u_s.shape[0])
    print("after target proj (u_t):", u_t.shape[0])
    print("colors_t shape:", colors_t.shape)
    print("min/max z_t:", z_t.min() if len(z_t) else None, z_t.max() if len(z_t) else None)

    color_splat, depth_splat, hit = forward_splat(u_t, v_t, z_t, colors_t, W, H, radius=args.splat_radius)
    print("hit sum:", hit.sum())
    depth_vis = normalize_depth_for_vis(depth_splat)

    # Overlay for sanity (target RGB + small dots)
    overlay = img_tgt.copy()
    for uu, vv in zip(u_t[::20], v_t[::20]):  # subsample for cleanliness
        cv2.circle(overlay, (int(round(uu)), int(round(vv))), 1, (255,0,0), -1)

    # Hole filling (very simple)
    if args.do_inpaint:
        # OpenCV inpaint expects a mask of holes (255 = hole)
        hole_mask = cv2.bitwise_not(hit)
        color_filled = cv2.inpaint(color_splat, hole_mask, 3, cv2.INPAINT_TELEA)
    else:
        color_filled = hole_fill(color_splat, hit, iterations=3)

    # Save
    outdir = f"Forward_splat_radius_{args.splat_radius}_out"
    print(f"Current Splat radius: {args.splat_radius}")
    os.makedirs(outdir, exist_ok=True)
    base = f"{args.src_frame}_to_{args.tgt_frame}"
    cv2.imwrite(f"{outdir}/novelview_overlay_{base}.png", overlay)
    cv2.imwrite(f"{outdir}/novelview_color_{base}.png", color_splat)
    cv2.imwrite(f"{outdir}/novelview_color_filled_{base}.png", color_filled)
    cv2.imwrite(f"{outdir}/novelview_depth_{base}.png", depth_vis)
    print("[OK] Wrote:",
          f"novelview_overlay_{base}.png, novelview_color_{base}.png, novelview_color_filled_{base}.png, novelview_depth_{base}.png")

    # --- Load ground truth image for comparison ---
    gt = img_tgt  # already loaded earlier

    # use color_filled OR color_splat depending on what you want
    pred = color_filled

    # make sure shapes match (resize GT if needed)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

    psnr = compute_psnr(pred, gt)
    ssim = compute_ssim(pred, gt)

    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Forward splatting (KITTI LiDAR -> novel view)")
    ap.add_argument("--kitti_root", default= "/home/sriramg/payalsaha/KITTI/dataset/")
    ap.add_argument("--sequence",  default= "00")
    ap.add_argument("--src_frame", default = "000000")
    ap.add_argument("--tgt_frame", default="000003")
    ap.add_argument("--poses_file", default="/home/sriramg/payalsaha/KITTI/dataset/poses/00.txt")
    ap.add_argument("--raw_calib_dir", default="/home/sriramg/payalsaha/KITTI/2011_10_03")
    ap.add_argument("--splat_radius", type=int, default=1, help="pixel radius of splat (1 or 2 works well)")
    ap.add_argument("--do_inpaint", type=int, default=0, help="0=dilation fill, 1=OpenCV inpaint")
    args = ap.parse_args()
    main(args)










