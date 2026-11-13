"""OLMo-3 full table: training plan/config + per-job code snapshot + sbatch.

Run once with --plan to write cartridges/{config,plan}.json; run with --snapshot
before every submission (fresh code_<n>, so a running job's code is never touched).
Each job: both ranks run remaining baseline units (no waiting), then the ratio
Cartridge worker (4 arms x 4 budgets, dynamic claim queue) until the 6h deadline.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/olmo3_full_table_20261113')
FAST = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/config.json'
METHODS = ['vanilla_cartridges', 'agnostic_dual_jsd', 'specific_sd', 'specific_dual_jsd']
PERCENTS = (5, 10, 15, 20)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(data, indent=1) + '\n'); tmp.replace(path)


def plan():
    base = json.loads((O / 'config.json').read_text())
    fast = json.loads(Path(FAST).read_text())
    requests = json.loads((O / 'requests.json').read_text())
    out = O / 'cartridges'
    bytes_per_token = 2 * 32 * 32 * 128 * 2  # OLMo-3-7B: 32 layers, 32 KV heads, head_dim 128, K+V, bf16
    rows = []
    for r in requests:
        for p in PERCENTS:
            prefix = int(r['whole_kv']['prefix_tokens'] * p // 100)
            kv = prefix * bytes_per_token
            rows.append(dict(r, whole_kv=dict(r['whole_kv'], path=None), index=len(rows), percent=p,
                prefix_tokens=prefix, memory=dict(native_kv_bytes=kv,
                training_peak_estimate_bytes=int(16e9 + 4 * kv + 12e9), estimate_only=True)))
    for r in rows:
        r.pop('worker_rank', None)
    write(out / 'plan.json', rows)
    config = {k: fast[k] for k in ('training', 'initialization_chunk_tokens', 'metric_policy', 'seed',
                                   'cache_budget_mode', 'ratio_denominator', 'maximum_kv_ratio')}
    config.update(workspace=str(ROOT), model=base['model'], dataset_root=base['dataset_root'],
        python=base['python'], python_default=base['python_default'], python_blackwell=base['python_blackwell'],
        runtime_cache_root=base['runtime_cache_root'], output=str(out), plan=str(out / 'plan.json'),
        runtime_code_root='SET_PER_JOB', protocol='olmo3-full-table-ratio-4arm-v1', methods=METHODS,
        percentages=list(PERCENTS), native_backbone=True, dynamic_work_queue=True,
        share_sd_jsd_warmup=False, retain_finished_checkpoints=False, workers=2,
        execution_window_hours=6, deadline_clock='first_gpu_allocation', campaign_clock_path='SET_PER_JOB',
        dual_view=dict(topk_per_view=20, target_mass=0.99, max_entries=40, prefill_tokens=1024),
        method_labels=dict(vanilla_cartridges='Vanilla Cartridges (agnostic data + SD)',
            agnostic_dual_jsd='Vanilla + DualJSD', specific_sd='Specific + SD',
            specific_dual_jsd='Contextual Cartridges (ours): pseudo-answer-guided data + DualJSD'),
        accuracy_guaranteed=False, benchmark_answers_used_for_training=False)
    config['training'] = dict(config['training'], sd_offload_saved_tensors=False)
    write(out / 'config.json', config)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    print(len(rows), 'cells')


def snapshot():
    n = 1 + max([int(p.name.split('_')[-1]) for p in O.glob('code_main_*')] or [0])
    code = O / f'code_main_{n}'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'runs', '*.log')
    for name in ('baselines', 'contextualized', 'experiments', 'tests'):
        shutil.copytree(ROOT / name, code / name, ignore=ignore, symlinks=True)
    shutil.copytree(ROOT / 'cartridges' / 'cartridges', code / 'upstream' / 'cartridges', ignore=ignore)
    tag = f'job{n}'
    cfg_path = O / 'cartridges' / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    cfg.update(runtime_code_root=str(code), campaign_clock_path=str(O / 'cartridges' / f'clock_{tag}.json'))
    cfg.pop('deadline_epoch', None)
    write(cfg_path, cfg)
    syn = json.loads((O / 'config.json').read_text())
    syn.update(runtime_code_root=str(code), campaign_clock_path=str(O / f'clock_synthesis_{tag}.json'),
               answer_instruction='nonempty_v2')
    syn.pop('deadline_epoch', None)
    write(O / 'config.json', syn)
    b = json.loads((O / 'baselines_config.json').read_text())
    b['runtime_code_root'] = str(code)
    write(O / 'baselines_config.json', b)
    q = str(ROOT)
    rank = f'''#!/bin/bash
# per-rank: pick the CUDA runtime for this GPU, run baselines (no waiting), then training
names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
PY={cfg['python_default']}
if echo "$names" | grep -qiE "blackwell|rtx pro 6000|b200"; then PY={cfg['python_blackwell']}; fi
CUDA=/apps/software/system/software/CUDA/12.8.0
export PYTHONPATH={code}:{q}/.deps:{code}/upstream:{code}/baselines/upstream/compaction
export CARTRIDGES_DIR={q}/cartridges CARTRIDGES_OUTPUT_DIR={O} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_HOME=$CUDA PATH=$(dirname $PY):{q}/.deps/toolchain/bin:$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}
export BASELINE_NO_WAIT=1
cd {code}
/usr/bin/python3 -u experiments/run_thirty_study.py --config {O}/config.json   # sources + OLMo synthesis (this rank's requests)
echo "synthesis exit $?"
cd {code}/baselines/upstream/compaction   # authors' AM configs are cwd-relative
$PY -u {code}/experiments/run_olmo_baselines.py --config {O}/baselines_config.json
cd {code}
echo "baselines exit $?"
# Training passes; later passes pick up requests whose synthesis finished on the other GPU.
for pass in $(seq 1 40); do
  /usr/bin/python3 -u experiments/run_ratio_retrain.py --config {cfg_path}
  code=$?; echo "training pass $pass exit $code"
  [ "$code" = 75 ] && break
  if [ -n "${{SLURM_JOB_END_TIME:-}}" ] && [ $(( SLURM_JOB_END_TIME - $(date +%s) )) -lt 900 ]; then break; fi
  sleep 300
done
'''
    (code / 'rank.sh').write_text(rank); (code / 'rank.sh').chmod(0o755)
    sbatch = f'''#!/bin/bash
#SBATCH --job-name=olmo-main
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu:rtx_pro_6000_blackwell|gpu:h200|gpu:b200
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --open-mode=append
#SBATCH --output={O}/cartridges/logs/job-%j.out
#SBATCH --error={O}/cartridges/logs/job-%j.err
set -uo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
srun --unbuffered --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=0 \\
  --output={O}/cartridges/logs/rank-%t-%j.out --error={O}/cartridges/logs/rank-%t-%j.err {code}/rank.sh
exit 0
'''
    (code / 'run_main.sbatch').write_text(sbatch)
    print(code / 'run_main.sbatch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    if args.plan:
        plan()
    if args.snapshot:
        snapshot()
