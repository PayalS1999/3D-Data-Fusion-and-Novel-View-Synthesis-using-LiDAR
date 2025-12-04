#!/bin/bash
#SBATCH --job-name=dsnerf_train_gpu
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/dsnerf_train_gpu_%j.out
#SBATCH --error=logs/dsnerf_train_gpu_%j.err

set -euo pipefail

echo "===== TRAIN START ====="
date
hostname
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "======================="

# ---- Conda env ----
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate nerfstudio

# ---- Paths ----
cd "$HOME/kitti_nerf_500_s8"
DATA_OUT=/home/sriramg/payalsaha/kitti_nerf_500_s8
RUN_OUT=/home/sriramg/payalsaha/kitti_nerf_500_s8/

# Train depth-nerfacto on GPU
ns-train depth-nerfacto \
  --data "$DATA_OUT" \
  --output-dir "$RUN_OUT" \
  --max-num-iterations 40000 \
  --viewer.quit-on-train-completion True \
  --pipeline.model.predict-normals False \
  2>&1 | tee training.log

echo "===== TRAIN DONE ====="
date