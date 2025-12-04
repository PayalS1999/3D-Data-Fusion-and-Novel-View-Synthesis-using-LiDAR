import json

# Input files
train_file = "/home/sriramg/payalsaha/kitti_nerf_scene/transforms_train.json"
test_file  = "/home/sriramg/payalsaha/kitti_nerf_scene/transforms_test.json"

# Output file
output_file = "/home/sriramg/payalsaha/kitti_nerf_scene/transforms.json"

# Load train
with open(train_file, "r") as f:
    train_data = json.load(f)

# Load test
with open(test_file, "r") as f:
    test_data = json.load(f)

# Use train metadata as base (assume identical except for frames)
meta = {k: train_data[k] for k in train_data if k != "frames"}

merged_frames = []

# Add train frames
for frame in train_data["frames"]:
    frame["split"] = "train"
    merged_frames.append(frame)

# Add test frames
for frame in test_data["frames"]:
    frame["split"] = "val"
    merged_frames.append(frame)

# Sort by image filename (optional but cleaner)
merged_frames.sort(key=lambda f: f["file_path"])

# Combine
output_json = meta
output_json["frames"] = merged_frames

# Save
with open(output_file, "w") as f:
    json.dump(output_json, f, indent=4)

print("Wrote merged transforms.json with split field.")
print(f"Total frames: {len(merged_frames)}")