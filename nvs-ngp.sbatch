#!/bin/bash
#SBATCH --job-name=instant_ngp
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/instant_ngp_train_%j.out
#SBATCH --error=logs/instant_ngp_train_%j.err

set -euo pipefail

echo "===== TRAIN START ====="
date
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "======================="
cd "$HOME/instant-ngp"
# rm -rf build/ || true

# ---- Conda env ----
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ngp

# ls -d /usr/local/cuda*

# Some clusters set CUDA_HOME; if not, you can echo it with: echo $CUDA_HOME
# Use CUDA 12.4
export CUDA_HOME=/usr/local/cuda-12.4
export CUDATOOLKIT_ROOT=$CUDA_HOME

# Make sure nvcc and libs are visible
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# (Optional but nice for CMake)
export CUDACXX="$CUDA_HOME/bin/nvcc"

# cmake -B build \
#  -DNGP_BUILD_WITH_GUI=OFF \
#  -DNGP_BUILD_WITH_CUDA=ON \
#  -DPython_EXECUTABLE="$(which python)"

# cmake -B build -DNGP_BUILD_WITH_GUI=OFF -DNGP_BUILD_WITH_CUDA=ON

# cmake --build build -j

# 2) Force system libstdc++ instead of conda’s
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 3) (Optional) still make sure CUDA is on the path if needed
export PATH=/usr/local/cuda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 4) Run instant-ngp
./build/instant-ngp --help
ls build/pyngp*.so
export PYTHONPATH="${PYTHONPATH:-}:/home/sriramg/payalsaha/instant-ngp/build"

python3 - << 'EOF'
import pyngp as ngp
print("pyngp loaded correctly")
EOF

echo "===== Running Train script ====="

cd "$HOME/nvs-kitti"
# python nerf_transform/ngp_train.py
python neural_rendering/ngp_render_val.py

# ---- Paths ----
DATA_OUT=/home/sriramg/payalsaha/kitti_ngp_scene_500_s5
RUN_OUT=/home/sriramg/payalsaha/kitti_ngp_scene_500_s5

# Train NGP on GPU
# ./build/instant-ngp \
#  --scene $DATA_OUT/transforms_train_ngp.json
echo "===== TRAIN DONE ====="
date