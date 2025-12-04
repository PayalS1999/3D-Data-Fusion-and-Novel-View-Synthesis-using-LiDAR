#!/bin/bash
#SBATCH --job-name=forward_splat
#SBATCH --partition=short        # uses cn2/cn5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # change if you want more/less
#SBATCH --mem=64G                # adjust if needed
#SBATCH --time=02:00:00          # 2 hours; adjust as per your needs
#SBATCH --output=logs/forward_splat-%j.out
#SBATCH --error=logs/forward_splat-%j.err

echo "===== SLURM JOB STARTED ====="

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" # or ~/.zshrc / cluster init file
conda activate lidar-nvs          # your env

# Go to project dir
cd ~/nvs-kitti                    # change if needed

# Run your script
python forward_splat_kitti.py \
 --src_frame "000011" \
 --tgt_frame "000017" \
 --splat_radius 3

echo "===== DONE ====="