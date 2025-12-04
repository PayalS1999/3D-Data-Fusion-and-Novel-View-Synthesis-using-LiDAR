import os
import time
import pyngp as ngp

scene = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/transforms_train_ngp.json"
snapshot_path = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5/kitti00_nerf.msgpack"
config = "/home/sriramg/payalsaha/instant-ngp/configs/nerf/base.json"

target_iters = 200000
log_every = 5


def main():
    # 1. Create testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    # 2. Load scene + network config explicitly
    testbed.load_training_data(scene)
    testbed.reload_network_from_file(config)

    testbed.shall_train = True

    print("Starting training...")
    t0 = time.time()

    last_step = -1
    while last_step < target_iters:
        # one optimization step over random rays
        testbed.frame()

        step = testbed.training_step
        # safety: if training_step never increases, bail out
        if step == last_step:
            print("WARNING: training_step not increasing → likely 0 valid rays / bad scene.")
            break
        last_step = step

        if step % log_every == 0:
            loss = float(testbed.loss)
            elapsed = time.time() - t0
            iters_per_sec = step / max(elapsed, 1e-6)
            print(f"[iter {step:7d}] loss={loss:.6f}, it/s={iters_per_sec:.1f}", flush=True)

    print("Training completed (or stopped). Saving snapshot...")
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    testbed.save_snapshot(snapshot_path, False)
    print("Saved:", snapshot_path)


if __name__ == "__main__":
    main()