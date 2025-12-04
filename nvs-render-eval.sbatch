#!/bin/bash
#SBATCH --job-name=nerf_render_eval
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/render_eval_%j.out
#SBATCH --error=logs/render_eval_%j.err

set -euo pipefail

echo "===== RENDER/EVAL START ====="
date
hostname
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================="

# ---- Conda env ----
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate nerfstudio

# ---- Paths (EDIT if needed) ----

cd "$HOME/kitti_nerf_500"
RUN_NUM=2025-12-01_130842
DATA_OUT=/home/sriramg/payalsaha/kitti_nerf_500
RUN=/home/sriramg/payalsaha/kitti_nerf_500_s8/kitti_nerf_500_s8/depth-nerfacto/$RUN_NUM

mkdir -p "$DATA_OUT/render_dsnerf_$RUN_NUM"

OUT_PATH=$DATA_OUT/render_dsnerf_$RUN_NUM

ns-render dataset \
  --load-config "$RUN/config.yml" \
  --split val \
  --output-path "$OUT_PATH/val_images"


ns-eval \
  --load-config "$RUN/config.yml" \
  --output-path "$OUT_PATH/val_metrics.json"


echo "===== DONE ====="
date