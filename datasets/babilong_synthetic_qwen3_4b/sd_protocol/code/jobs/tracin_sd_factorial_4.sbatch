#!/bin/bash
#SBATCH --job-name=tracin-selective-sd
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu,gpu_rtx6000,scavenge_gpu,gpu_h200,gpu_h100,gpu_b200
# TracIn needs two native-KV gradient passes in addition to the 4B backbone;
# 24/32 GB cards OOM even though ordinary SD itself fits.  Keep only >=48 GB.
#SBATCH --constraint=gpu:rtx_pro_6000_blackwell|gpu:l40s|gpu:h200|gpu:h100|gpu:b200
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/da839/Query_Cart/runs/logs/tracin-selective-sd-%A_%a.out
#SBATCH --error=/home/da839/Query_Cart/runs/logs/tracin-selective-sd-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --gpu-bind=single:1 /usr/bin/python3 -u \
  /home/da839/Query_Cart/experiments/dispatch_syn_sd_factorial.py \
  --config "$1" --rank "$SLURM_ARRAY_TASK_ID"
