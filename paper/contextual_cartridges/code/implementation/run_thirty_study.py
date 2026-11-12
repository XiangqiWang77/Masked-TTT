"""Separate reranking, vLLM synthesis, and native SD processes within one job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_figure4_contextual import load_cases, write


def configured_stages(config):
    stages = tuple(config.get('execution_stages', ('sources', 'synthesis', 'train')))
    if stages not in (('sources', 'synthesis'), ('sources', 'synthesis', 'train')):
        raise ValueError('Require source preparation, synthesis, and optionally training')
    if config.get('synthesis_only') and stages != ('sources', 'synthesis'):
        raise ValueError('A synthesis-only allocation must not launch training')
    return stages


def stage(config, name, rank, deadline):
    # User-requested generation revisions can be prepared while the separate
    # source-preparation process finishes, without dropping its GPU allocation.
    ready = config.get('generation_revision_ready_path')
    if name == 'synthesis' and ready:
        while not Path(ready).exists():
            if time.time() >= deadline-30:
                raise TimeoutError('Generation revision was not released before deadline')
            time.sleep(1)
        config = json.loads((Path(config['output'])/'config.json').read_text())
    if name not in configured_stages(config):
        raise ValueError('This stage is not authorized by the run configuration')
    from experiments.thirty_study_data import prepare_sources, generate
    requests = [r for r in json.loads(Path(config['requests']).read_text()) if r['worker_rank'] == rank]
    cases = {d: {c['id']: c for c in load_cases(d, config['dataset_root'])}
             for d in {r['dataset'] for r in requests}}
    for row in requests:
        case = cases[row['dataset']][row['case']]
        for field, expected in (('document', row['document_sha256']), ('prompt', row['prompt_sha256'])):
            if hashlib.sha256(case[field].encode()).hexdigest() != expected:
                raise ValueError('Evaluation source changed since planning')
    write(Path(config['output'])/f'stage-{rank}.json', dict(stage=name, started_epoch=time.time()))
    if name == 'sources':
        prepare_sources(config, requests, cases, deadline)
    elif name == 'synthesis':
        generate(config, requests, cases, deadline)
    elif name == 'train':
        from experiments.run_ratio_retrain import worker
        return worker(config, deadline, rank)
    else:
        raise ValueError('Unknown job stage')
    return 0


def dispatch(config_path):
    from experiments.gpu_campaign_clock import start_execution_window
    from experiments.dispatch_figure4_contextual import runtime_for_gpu
    config = start_execution_window(config_path)
    rank = int(os.environ['SLURM_PROCID'])
    if rank not in (0, 1) or config['workers'] != 2:
        raise ValueError('Require two single-GPU ranks')
    deadline = config['deadline_epoch']
    if os.environ.get('SLURM_JOB_END_TIME', '').isdigit():
        deadline = min(deadline, float(os.environ['SLURM_JOB_END_TIME']))
    deadline -= 180
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    python, runtime = runtime_for_gpu(config, names)
    repo, workspace = Path(config['runtime_code_root']), Path(config['workspace'])
    cuda = Path('/apps/software/system/software/CUDA/12.8.0')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{repo}:{workspace}/.deps:{repo}/upstream',
        CARTRIDGES_DIR=str(workspace/'cartridges'), CARTRIDGES_OUTPUT_DIR=config['output'],
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled', RETRAIN_DEADLINE=str(deadline),
        CUDA_HOME=str(cuda), CUDA_ROOT=str(cuda), CUDA_PATH=str(cuda),
        VLLM_WORKER_MULTIPROC_METHOD='spawn')
    env['PATH'] = f'{Path(python).parent}:{workspace}/.deps/toolchain/bin:{cuda}/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH'] = f'{cuda}/targets/x86_64-linux/lib:'+env.get('LD_LIBRARY_PATH','')
    for name, sub in {'TORCHINDUCTOR_CACHE_DIR':'inductor', 'TRITON_CACHE_DIR':'triton',
        'TORCH_EXTENSIONS_DIR':'extensions', 'CUDA_CACHE_PATH':'cuda', 'XDG_CACHE_HOME':'xdg'}.items():
        path = Path(config['runtime_cache_root'])/runtime/sub
        path.mkdir(parents=True, exist_ok=True); env[name] = str(path)
    child, stopping = None, False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps(dict(event='thirty_dispatch', rank=rank, gpu=names.strip(),
        stages=configured_stages(config), python=python, deadline=deadline)), flush=True)
    for name in configured_stages(config):
        while not stopping and time.time() < deadline-30:
            child = subprocess.Popen([python, '-u', str(Path(__file__).resolve()),
                '--config', config_path, '--stage', name], env=env, cwd=repo, start_new_session=True)
            try:
                code = child.wait(timeout=max(1, deadline-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                return 75
            if code == 76 and name == 'train':
                continue  # same established worker recovery; failed unit is not retried
            if code != 0:
                return code
            break
        if stopping or time.time() >= deadline-30:
            return 75
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('sources', 'synthesis', 'train'))
    args = parser.parse_args()
    if args.stage:
        config = json.loads(Path(args.config).read_text())
        try:
            code = stage(config, args.stage, int(os.environ['SLURM_PROCID']), float(os.environ['RETRAIN_DEADLINE']))
        except TimeoutError:
            code = 75
        except Exception as error:
            traceback.print_exc()
            write(Path(config['output'])/'failures'/f"stage-{args.stage}-{os.environ['SLURM_PROCID']}.json",
                  dict(stage=args.stage, error=str(error), epoch=time.time()))
            code = 1
        raise SystemExit(code)
    raise SystemExit(dispatch(args.config))
