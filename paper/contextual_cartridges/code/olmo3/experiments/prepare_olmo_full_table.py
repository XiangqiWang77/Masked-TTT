"""OLMo-3-7B-Instruct full-table campaign: plan, Whole-KV sizes, code snapshot, sbatch.

Stage 1 (this file's sbatch): reranker sources + OLMo vLLM synthesis (15 agnostic,
15 pseudo-answer-guided specific trajectories per request). No training here.
Requests: LongHealth 200 (patients 1-10, all questions), QASPER 50, MTOB-KE 50,
interleaved so a partial run covers every dataset.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_figure4_contextual import load_cases, write  # noqa: E402

MODEL = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/models/Olmo-3-7B-Instruct'
DATA = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_cartridges_figure4_datasets_20261110'
SYNTH_REF = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/synthetic_threeway_min15_20261112/config.json'
COUNTS = {'longhealth': 200, 'qasper': 50, 'mtob_ke': 50}
PATTERN = ['longhealth', 'longhealth', 'qasper', 'longhealth', 'longhealth', 'mtob_ke']


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def whole_kv(tokenizer, document, prompts):
    """Same rule as baselines.hf_methods.make_source_cache, tokenizer only."""
    from baselines.protocol import bounded_context_ids
    from evaluation.utils import format_context
    maximum = int(json.loads((Path(MODEL) / 'config.json').read_text())['max_position_embeddings'])
    q = lambda p: len(tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=True,
                                                    add_generation_prompt=True))
    reserve = max(q(p) for p in prompts) + 1024
    ids = lambda t: tokenizer.encode(t, add_special_tokens=False)
    raw = ids(document)
    wrapper = len(ids(format_context(tokenizer, '', model_name=MODEL)))
    retained, truncated = bounded_context_ids(raw, maximum - reserve - wrapper - 16)
    text = tokenizer.decode(retained, skip_special_tokens=False)
    prefix = len(ids(format_context(tokenizer, text, model_name=MODEL)))
    return dict(document_tokens=len(raw), retained_document_tokens=len(retained), truncated=truncated,
                truncation_policy='native_window_head_tail', native_window_tokens=maximum,
                prefix_tokens=prefix, full_document=not truncated, measured_by='olmo3_tokenizer_only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    chosen = {}
    for dataset, count in COUNTS.items():
        cases = load_cases(dataset, DATA)
        chosen[dataset] = cases[:count]
        assert len(chosen[dataset]) == count, (dataset, len(cases))
    sizes = {}
    for dataset, cases in chosen.items():
        by_doc = {}
        for c in cases:
            by_doc.setdefault(sha(c['document']), []).append(c)
        for key, group in by_doc.items():
            sizes[key] = whole_kv(tokenizer, group[0]['document'], [c['prompt'] for c in group])
    order, cursor = [], {d: 0 for d in COUNTS}
    while len(order) < sum(COUNTS.values()):
        for d in PATTERN:
            if cursor[d] < COUNTS[d]:
                order.append(chosen[d][cursor[d]]); cursor[d] += 1
                order[-1] = (d, order[-1])
    requests = []
    for index, (dataset, case) in enumerate(order):
        key = sha(case['document'])
        requests.append(dict(dataset=dataset, case=case['id'], position=index, document_key=key[:20],
            document_sha256=key, prompt_sha256=sha(case['prompt']), whole_kv=sizes[key],
            request_index=index, worker_rank=index % 2, synthetic_questions=15,
            synthesis_manifest=str(out / 'synthetic' / dataset / case['id'] / 'pair.json')))
    write(out / 'requests.json', requests)
    ref = json.loads(Path(SYNTH_REF).read_text())
    config = {k: ref[k] for k in ('workspace', 'python', 'python_default', 'python_blackwell',
                                   'runtime_cache_root', 'reranker_model', 'max_evidence_attempts', 'seed')}
    config.update(model=MODEL, dataset_root=DATA, output=str(out), requests=str(out / 'requests.json'),
        runtime_code_root=str(out / 'code'), protocol='olmo3-full-table-synthesis-v1',
        workers=2, execution_window_hours=6, deadline_clock='first_gpu_allocation',
        campaign_clock_path=str(out / 'execution_clock_synthesis.json'),
        execution_stages=['sources', 'synthesis'], synthesis_only=True, include_evidence_only=False,
        reuse_existing_questions=False, synthetic_questions_per_request_per_arm=15,
        minimum_questions_per_request_per_arm=15, generation_parallel_requests=4,
        generation_max_sequences=32, generation_gpu_memory_utilization=0.6, reranker_batch_size=8,
        generation_backend='vllm_0.12.0_batched_raw_teacher_probabilities',
        expected_requests=len(requests), counts=COUNTS, automatic_training=False,
        accuracy_guaranteed=False, benchmark_answers_used=False,
        note='Contextual Cartridges = pseudo-answer-guided specific questions + DualJSD (trained later).')
    write(out / 'config.json', config)
    code = out / 'code'
    if code.exists():
        raise SystemExit(f'{code} exists; never overwrite a snapshot')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in ('baselines', 'contextualized', 'experiments', 'tests'):
        shutil.copytree(ROOT / name, code / name, ignore=ignore, symlinks=True)
    shutil.copytree(ROOT / 'cartridges' / 'cartridges', code / 'upstream' / 'cartridges', ignore=ignore)
    (out / 'logs').mkdir(exist_ok=True)
    sbatch = f'''#!/bin/bash
#SBATCH --job-name=olmo-synth
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu:rtx_pro_6000_blackwell|gpu:h200|gpu:b200
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --open-mode=append
#SBATCH --output={out}/logs/job-%j.out
#SBATCH --error={out}/logs/job-%j.err
set -uo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd {code}
srun --unbuffered --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=0 \\
  --output={out}/logs/worker-%t-%j.out --error={out}/logs/worker-%t-%j.err \\
  /usr/bin/python3 {code}/experiments/run_thirty_study.py --config {out}/config.json
status=$?
[[ "$status" == 75 ]] && {{ echo 'window ended; partial data retained'; exit 0; }}
exit "$status"
'''
    (out / 'run_6h.sbatch').write_text(sbatch)
    summary = {d: dict(requests=COUNTS[d], prefix_tokens=sorted({r['whole_kv']['prefix_tokens'] for r in requests
               if r['dataset'] == d})[:5], truncated=any(r['whole_kv']['truncated'] for r in requests if r['dataset'] == d))
               for d in COUNTS}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
