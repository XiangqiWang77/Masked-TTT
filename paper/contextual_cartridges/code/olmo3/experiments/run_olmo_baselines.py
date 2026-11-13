"""OLMo-3 frozen baselines for the full table, reusing the audited Qwen-era routines.

Methods: icl (Whole KV), evidence_teacher (Evidence ICL), snapkv, fast_kv (official
AM-HighestAttnKeys-fast, text-chunked). Units = (method, dataset, document) over the
campaign's requests; two ranks share one dynamic claim queue. Ratio budgets are
5/10/15/20% of the measured OLMo Whole-KV prefix (stored-byte ceilings as before).
Evidence ICL waits for the request's synthesis (its spans are the union of the
source spans behind the saved query-specific targets) and is retried each pass.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_figure4_contextual import load_cases, write  # noqa: E402

PERCENTS = (5, 10, 15, 20)


def units(config):
    requests = json.loads(Path(config['requests']).read_text())
    groups = {}
    for r in requests:
        groups.setdefault((r['dataset'], r['document_key']), []).append(r)
    out = []
    for method in config['methods']:
        for (dataset, key), rows in groups.items():
            whole = rows[0]['whole_kv']
            out.append(dict(method=method, dataset=dataset, document_key=key,
                case_ids=[r['case'] for r in rows], first_position=min(r['position'] for r in rows),
                whole_kv=dict(whole, path=None),
                ratio_budgets=[dict(percent=p, prefix_tokens=int(whole['prefix_tokens'] * p // 100))
                               for p in PERCENTS] if method in ('snapkv', 'fast_kv') else None))
    rank = {m: i for i, m in enumerate(config['methods'])}
    out.sort(key=lambda u: (rank[u['method']], u['first_position']))
    for i, u in enumerate(out):
        u['index'] = i
        if u['ratio_budgets'] is None:
            del u['ratio_budgets']
    return out


def evidence_record(config, dataset, case):
    """Adapter: synthesis specific.json -> baselines.evidence_context record."""
    spec = Path(config['output']) / 'synthetic' / dataset / case['id'] / 'specific.json'
    manifest = spec.with_name('pair.json')
    if not manifest.exists():
        return None
    data = json.loads(spec.read_text())
    target = Path(config['evidence_source_root']) / dataset / f"{case['id']}.json"
    if not target.exists():
        write(target, dict(dataset=dataset, case=case['id'], query=case['query'],
            examples=[dict(evidence=e['evidence']) for e in data['examples']],
            source='olmo3 synthesis specific.json (pseudo-answer-guided)', benchmark_answer_used=False))
    return target


def done(config, unit):
    from baselines.protocol import result_path
    if 'ratio_budgets' in unit:
        return all(result_path(config['output'], unit, b['prefix_tokens'], c).exists()
                   for b in unit['ratio_budgets'] for c in unit['case_ids'])
    return all(result_path(config['output'], unit, None, c).exists() for c in unit['case_ids'])


def worker(config, rank, deadline):
    import torch
    from baselines.dispatch import claim
    from baselines.hf_methods import FrozenBaselines
    from baselines.run_unit import frozen_methods
    from baselines.ratio_runner import run_ratio_unit
    out = Path(config['output'])
    plan = units(config)
    write(out / 'baseline_plan.json', plan)
    datasets, runtime = {}, None
    while time.time() < deadline - 120:
        progressed, waiting = False, False
        for unit in plan:
            if time.time() >= deadline - 120:
                return 75
            failure = out / 'failures' / f"baseline-{unit['index']:04d}.json"
            if failure.exists() or done(config, unit):
                continue
            lock = claim(out / 'claims', 100000 + unit['index'])
            if lock is None:
                continue
            try:
                if done(config, unit):
                    continue
                if unit['dataset'] not in datasets:
                    datasets[unit['dataset']] = {c['id']: c for c in load_cases(unit['dataset'], config['dataset_root'])}
                cases = [datasets[unit['dataset']][c] for c in unit['case_ids']]
                if unit['method'] == 'evidence_teacher':
                    ready = [c for c in cases if evidence_record(config, unit['dataset'], c)]
                    if len(ready) < len(cases):
                        waiting = True
                    cases = ready
                    if not cases:
                        continue
                if runtime is None:
                    runtime = FrozenBaselines(config, deadline - 60)
                directory = out / 'artifacts' / unit['method'] / unit['dataset'] / unit['document_key']
                directory.mkdir(parents=True, exist_ok=True)
                began = time.time()
                if unit['method'] in ('snapkv', 'fast_kv'):
                    run_ratio_unit(config, unit, cases, directory, deadline - 60, runtime)
                else:
                    frozen_methods(config, unit, cases, directory, deadline - 60, runtime)
                print(json.dumps(dict(event='baseline_unit_done', rank=rank, method=unit['method'],
                    dataset=unit['dataset'], cases=len(cases), seconds=round(time.time() - began, 1))), flush=True)
                progressed = True
            except TimeoutError:
                return 75
            except Exception as error:
                traceback.print_exc()
                write(failure, dict(unit={k: v for k, v in unit.items() if k != 'whole_kv'},
                                    rank=rank, error=str(error), epoch=time.time()))
                gc.collect(); torch.cuda.empty_cache()
            finally:
                lock.close()
        if not progressed:
            if not waiting or os.environ.get('BASELINE_NO_WAIT'):
                return 0  # leftover evidence units are retried by the next job
            time.sleep(60)  # evidence waits on synthesis
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    # Each allocation is its own window: the job end time is the deadline.
    end = float(os.environ.get('SLURM_JOB_END_TIME', 0) or 0) or time.time() + float(os.environ.get('WINDOW_SECONDS', 3600))
    raise SystemExit(worker(cfg, int(os.environ.get('SLURM_PROCID', 0)), end - 120))
