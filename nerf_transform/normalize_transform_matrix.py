#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

IN_JSON  = Path("/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/transforms_test.json")
OUT_JSON = Path("/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/transforms_test_ngp.json")

def main():
    with open(IN_JSON, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    # 1) Collect camera centers
    centers = []
    for fr in frames:
        T = np.array(fr["transform_matrix"], dtype=np.float64)  # 4x4
        c = T[:3, 3]
        centers.append(c)
    centers = np.stack(centers, axis=0)  # (N,3)

    # 2) Compute mean center
    center_mean = centers.mean(axis=0)   # (3,)
    print("Center mean:", center_mean)

    # 3) Compute radius and scale so that max distance ≈ 1.0
    dists = np.linalg.norm(centers - center_mean, axis=1)
    max_dist = dists.max()
    scale = 1.0 / max_dist
    print("Max dist:", max_dist, " -> scale:", scale)

    # 4) Apply recenter + scale to each transform
    for fr in frames:
        T = np.array(fr["transform_matrix"], dtype=np.float64)
        # recenter translation
        T[:3, 3] = (T[:3, 3] - center_mean) * scale
        fr["transform_matrix"] = T.tolist()

    # Optional: store scale/offset just for reference
    data["ngp_center_mean"] = center_mean.tolist()
    data["ngp_scale"] = float(scale)

    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print("[OK] Wrote", OUT_JSON)

if __name__ == "__main__":
    main()