#!/usr/bin/env python3
import os
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import pyngp as ngp

SCENE_TRAIN = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/transforms_train_ngp.json"
SCENE_VAL   = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/transforms_test_ngp.json"
SNAPSHOT    = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/kitti00_nerf.msgpack"
CONFIG      = "/home/sriramg/payalsaha/instant-ngp/configs/nerf/base.json"

OUT_DIR     = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/renders_ngp_val"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load val transforms (camera poses + intrinsics)
    with open(SCENE_VAL, "r") as f:
        meta = json.load(f)

    frames = meta["frames"]
    W, H = int(meta["w"]), int(meta["h"])

    print(f"Val frames: {len(frames)}, resolution: {W}x{H}")

    # Build testbed with the *training* scene + config, then load snapshot
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf, SCENE_TRAIN, CONFIG)
    testbed.shall_train = False
    testbed.load_snapshot(SNAPSHOT)

    # Make sure render resolution matches dataset
    try:
        testbed.set_render_res(W, H)
    except AttributeError:
        # older builds just use width/height from camera; safe to ignore
        pass

    for i, frame in enumerate(frames):
        cam2world = np.array(frame["transform_matrix"], dtype=np.float32)

        # instant-ngp expects a *camera matrix*. For the Python API this is
        # usually camera-to-world:
        #   testbed.camera_matrix = cam2world
        #
        # If renders look flipped / wrong, you can try inv(cam2world) instead.
        cam_matrix = cam2world[:3, :]
        testbed.camera_matrix = cam_matrix

        # Render RGB (H,W,4) in [0,1]
        rgb = testbed.render(W, H, spp=4, linear=False)
        rgb = np.clip(rgb[..., :3] * 255.0, 0, 255).astype(np.uint8)

        name = Path(frame["file_path"]).name  # e.g. "002345.png"
        out_path = os.path.join(OUT_DIR, name)
        imageio.imwrite(out_path, rgb)

        print(f"[{i}/{len(frames)}] saved {out_path}")

    print("Done rendering val set.")

if __name__ == "__main__":
    main()