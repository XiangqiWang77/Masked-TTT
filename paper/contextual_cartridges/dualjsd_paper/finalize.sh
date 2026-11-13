#!/bin/bash
# Re-aggregate final results, redraw figures, regenerate numbers/tables, recompile the paper.
set -euo pipefail
Q=/home/da839/Query_Cart
W=$Q/runs/dualjsd_paper_20261113
PY=/home/da839/.conda/envs/TTT/bin/python
srun -p devel -A pi_mg269 -c 2 --mem=16G -t 0:30:00 --chdir=$Q bash -c \
  "export CARTRIDGES_DIR=$Q/cartridges CARTRIDGES_OUTPUT_DIR=/tmp/x PYTHONPATH=$Q:$Q/.deps:$Q/cartridges; $PY $Q/runs/full_comparison_20261112/aggregate.py" > $W/aggregate.log 2>&1
cd $W
$PY make_figures.py > make_figures.log
python3 gen_tex.py
tectonic -X compile --keep-logs main.tex > tectonic.log 2>&1
grep "Output written" main.log
