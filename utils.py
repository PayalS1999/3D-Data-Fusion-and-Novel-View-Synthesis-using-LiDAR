import numpy as np

def load_velodyne_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts

def project_points(P, XYZ_cam):
    """
    P: (3,4) camera projection matrix
    XYZ_cam: (4,N) points in the camera frame (homogeneous)
    Returns u,v,z arrays and a valid mask (Z>0 and projected inside image later)
    """
    uvw = P @ XYZ_cam
    z = uvw[2,:]
    mask = z>1e-6
    u = uvw[0,mask] / z[mask]
    v = uvw[1,mask] / z[mask]
    z = z[mask]
    return u,v,z, mask


import numpy as np
import cv2

# -----------------------------
# PSNR
# -----------------------------
def compute_psnr(img1, img2):
    """Both inputs uint8 images, same shape."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# -----------------------------
# SSIM (simple luminance+contrast OpenCV variant)
# -----------------------------
def compute_ssim(img1, img2):
    """Returns SSIM in [0,1]. Both images must be uint8, same shape."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = 6.5025
    C2 = 58.5225

    img1_blur = cv2.GaussianBlur(img1, (11, 11), 1.5)
    img2_blur = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1 = img1_blur
    mu2 = img2_blur

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11,11), 1.5) - mu1 * mu1
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11,11), 1.5) - mu2 * mu2
    sigma12   = cv2.GaussianBlur(img1 * img2, (11,11), 1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))