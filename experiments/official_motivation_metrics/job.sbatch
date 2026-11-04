#!/bin/bash
#SBATCH --job-name=qc-official-motivation
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_devel,gpu,gpu_rtx6000,gpu_h100,gpu_h200,gpu_b200,scavenge_gpu
#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:rtx_pro_6000_blackwell|gpu:h100|gpu:h200|gpu:b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/official_motivation_metrics_20261104/logs/%j.out
#SBATCH --error=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/official_motivation_metrics_20261104/logs/%j.err
set -euo pipefail
unset OPENAI_API_KEY OPENAI_API_KEY_FILE
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
REPO=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/github_publications/gqa_head_gain_20261104/repo
RUNTIME=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102/query_cartridges_20261103/runtime
OUTPUT=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/official_motivation_metrics_20261104
export PYTHONPATH="$RUNTIME/src:$REPO"
export MPLCONFIGDIR="$OUTPUT/matplotlib"
PYTHON=/home/da839/.conda/envs/TTT/bin/python
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'B200|Blackwell'; then
    PYTHON=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/envs/ttt-cu128/bin/python
fi
cd "$REPO"
"$PYTHON" -B -m pytest -q tests/test_official_motivation_metrics.py
set +e
srun --kill-on-bad-exit=1 "$PYTHON" -B -u experiments/official_motivation_metrics/run.py --plan "$OUTPUT/plan.json" --output "$OUTPUT"
RUN_STATUS=$?
"$PYTHON" -B experiments/official_motivation_metrics/plot.py --plan "$OUTPUT/plan.json" --output "$OUTPUT"
PLOT_STATUS=$?
set -e
if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi
exit "$PLOT_STATUS"
