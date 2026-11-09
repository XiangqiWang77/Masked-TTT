#!/bin/bash
#SBATCH --job-name=tracin-sd-figures
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/da839/Query_Cart/runs/logs/tracin-sd-figures-%j.out
#SBATCH --error=/home/da839/Query_Cart/runs/logs/tracin-sd-figures-%j.err
set -euo pipefail

config="$1"
root="$2"
export PYTHONPATH=/home/da839/Query_Cart:/home/da839/Query_Cart/.deps:/home/da839/Query_Cart/cartridges
export CARTRIDGES_DIR=/home/da839/Query_Cart/cartridges
export CARTRIDGES_OUTPUT_DIR="$root"

/home/da839/.conda/envs/TTT/bin/python -u -m experiments.report_syn_sd_factorial \
  --config "$config"

/home/da839/.conda/envs/TTT/bin/python -u -m experiments.plot_tracin_sd_factorial \
  --report "$root/report/accuracy.json" \
  --experiment-root "$root" \
  --output "$root/report/final_figures"
