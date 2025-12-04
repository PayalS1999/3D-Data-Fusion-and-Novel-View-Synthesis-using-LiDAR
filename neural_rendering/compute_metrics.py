import os
import numpy as np
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

GT_DIR = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/images"
PRED_DIR = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/renders_ngp_val"

def compute_psnr_ssim(gt, pred):
    # both assumed in [0,1]
    data_range = 1.0
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=data_range)
    return psnr, ssim

def main():
    pred_files = sorted([f for f in os.listdir(PRED_DIR)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    psnrs, ssims = [], []

    for f in pred_files:
        gt_path = os.path.join(GT_DIR, f)
        pred_path = os.path.join(PRED_DIR, f)

        if not os.path.isfile(gt_path):
            print(f"[skip] GT missing for {f}: {gt_path}")
            continue

        gt = imageio.imread(gt_path).astype(np.float32) / 255.0
        pred = imageio.imread(pred_path).astype(np.float32) / 255.0

        if gt.shape != pred.shape:
            # just in case
            print(f"[warn] shape mismatch for {f}, gt={gt.shape}, pred={pred.shape}, resizing pred to gt")
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        psnr, ssim = compute_psnr_ssim(gt, pred)
        psnrs.append(psnr)
        ssims.append(ssim)

        print(f"{f}: PSNR={psnr:.3f}, SSIM={ssim:.4f}")

    if psnrs:
        print("\n==== Final Metrics over", len(psnrs), "images ====")
        print(f"Mean PSNR: {np.mean(psnrs):.3f}")
        print(f"Mean SSIM: {np.mean(ssims):.4f}")
    else:
        print("No pairs evaluated — check your GT_DIR and PRED_DIR paths.")

if __name__ == "__main__":
    main()