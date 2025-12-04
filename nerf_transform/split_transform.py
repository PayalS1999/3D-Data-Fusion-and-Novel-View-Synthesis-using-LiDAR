import json
from pathlib import Path

# Path to your existing transforms.json
ROOT = "/home/sriramg/payalsaha/kitti_ngp_scene_500_s5"
input_path = Path(f"{ROOT}/transforms.json")

with open(input_path, "r") as f:
    data = json.load(f)

frames = data["frames"]

train_frames = []
test_frames  = []

for idx, frame in enumerate(frames):
    # Every 10th frame → TEST set
    if idx % 20 == 0:
        test_frames.append(frame)
    else:
        train_frames.append(frame)

print(f"Total frames: {len(frames)}")
print(f"Train frames: {len(train_frames)}")
print(f"Test frames:  {len(test_frames)}")

# Write train file
train_json = {
    key: data[key] for key in data if key != "frames"
}
train_json["frames"] = train_frames

with open(f"{ROOT}/transforms_train.json", "w") as f:
    json.dump(train_json, f, indent=4)

# Write test file
test_json = {
    key: data[key] for key in data if key != "frames"
}
test_json["frames"] = test_frames

with open(f"{ROOT}/transforms_test.json", "w") as f:
    json.dump(test_json, f, indent=4)

print("Wrote transforms_train.json and transforms_test.json")