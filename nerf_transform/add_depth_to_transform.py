import json
from pathlib import Path

scene_root = Path("/home/sriramg/payalsaha/kitti_depth_nerf_scene")
tf_path = scene_root / "transforms.json"

with tf_path.open("r") as f:
    data = json.load(f)

for frame in data["frames"]:
    fp = frame["file_path"]  # e.g. "images/000007.png"
    stem = Path(fp).stem     # "000007"
    frame["depth_file_path"] = f"depth/{stem}.png"

with tf_path.open("w") as f:
    json.dump(data, f, indent=4)

print("Updated transforms.json with depth_file_path for all frames.")