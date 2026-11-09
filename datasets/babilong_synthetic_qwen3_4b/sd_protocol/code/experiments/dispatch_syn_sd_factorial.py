"""Dispatch the configured native-KV SD experiment inside the existing GPU array."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--rank', required=True, type=int)
    a = p.parse_args(); config = json.loads(Path(a.config).read_text())
    end = float(config['deadline_epoch'])
    slurm_end = os.environ.get('SLURM_JOB_END_TIME', '')
    if slurm_end.isdigit():
        end = (float(slurm_end) if config.get('deadline_from_slurm_allocation')
               else min(end, float(slurm_end)))
    if time.time() >= end-60: return 75
    names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).strip()
    blackwell = any(name in names.lower() for name in ('blackwell', 'b200', 'b300'))
    python = config['python_blackwell'] if blackwell else config['python_default']
    repo = Path('/home/da839/Query_Cart'); env = os.environ.copy()
    cuda_home = Path('/apps/software/system/software/CUDA/12.8.0')
    if not (cuda_home/'bin/nvcc').is_file():
        raise FileNotFoundError(f'CUDA compiler is unavailable: {cuda_home}/bin/nvcc')
    env['PATH'] = (f'{Path(python).parent}:{repo}/.deps/toolchain/bin:'
                   f'{cuda_home}/bin:'+env.get('PATH',''))
    env['CUDA_HOME'] = str(cuda_home)
    env['CUDA_ROOT'] = str(cuda_home)
    env['CUDA_PATH'] = str(cuda_home)
    env['LD_LIBRARY_PATH'] = (f'{cuda_home}/targets/x86_64-linux/lib:'
                              + env.get('LD_LIBRARY_PATH',''))
    cache_root=Path(config.get('runtime_cache_root',str(Path(config['output'])/'runtime_caches')))
    for name,directory in {'VLLM_CACHE_ROOT':'vllm','TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TRITON_CACHE_DIR':'triton','TORCH_EXTENSIONS_DIR':'extensions',
        'CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
        path=cache_root/directory;path.mkdir(parents=True,exist_ok=True);env[name]=str(path)
    env.update(PYTHONPATH=f'{repo}:{repo}/.deps:{repo}/cartridges', CARTRIDGES_DIR=str(repo/'cartridges'),
        CARTRIDGES_OUTPUT_DIR=config['output'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', PAIR_DEADLINE_EPOCH=str(end-120))
    print(json.dumps({'event': 'syn_sd_factorial_dispatch', 'rank': a.rank, 'gpu': names,
        'python': python, 'paired_cases': config['paired_cases'], 'matrix': config['matrix'],
        'deadline_epoch': end, 'smoke_test': False,
        'production_phase': config.get('production_phase', 'SD')}), flush=True)

    def run_child(script, path):
        child = subprocess.Popen([python, '-u', str(repo/'experiments'/script),
            '--config', str(path), '--rank', str(a.rank)], env=env, start_new_session=True)
        def stop(*_):
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try: return child.wait(timeout=max(1, end-time.time()-5))
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait(); return 75
        finally:
            # Clean up any engine subprocesses before the next GPU phase.
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass

    if config.get('sd_run_config'):
        return run_child('run_syn_sd_factorial.py',config['sd_run_config'])
    if config.get('vanilla_generation_config'):
        generation_path = Path(config['vanilla_generation_config'])
        status = run_child('run_vanilla_matched_1000.py', generation_path)
        if status not in (0,75): return status
        generation = json.loads(generation_path.read_text())
        full_config = Path(config['full1000_sd_config'])
        print(json.dumps({'event': 'waiting_for_full1000_handoff', 'rank': a.rank}), flush=True)
        while time.time() < min(end-120, generation['deadline_epoch']+600):
            ready = Path(generation['output'])/'dataset_complete.json'
            if full_config.is_file() and ready.is_file() and json.loads(ready.read_text()).get('complete'):
                return run_child('run_syn_sd_factorial.py', full_config)
            time.sleep(10)
        return 75
    return run_child('run_syn_sd_factorial.py', config.get('sd_run_config', a.config))


if __name__ == '__main__': raise SystemExit(main())
