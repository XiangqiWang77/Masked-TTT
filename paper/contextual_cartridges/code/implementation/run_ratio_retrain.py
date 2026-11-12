"""One two-GPU development job, fixed synthetic data, paired SD/JSD-SD arms."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.retrain_plan import METHODS, sha_file, matched_updates
from experiments.run_figure4_contextual import load_cases, write


def budget_label(row):
    return f"p{row['requested_prefix_tokens']}" if 'requested_prefix_tokens' in row else f"r{row['percent']:02d}"


def result_path(root, row, method):
    return Path(root)/'results'/method/row['dataset']/budget_label(row)/f"{row['case']}.json"


def complete(root, row):
    return all(result_path(root, row, method).exists() for method in METHODS)


def load_targets(row):
    """Never invokes a generator or a Teacher model; files must match the plan."""
    from cartridges.structs import read_conversations
    if row.get('synthesis_manifest'):
        manifest = json.loads(Path(row['synthesis_manifest']).read_text())
        if not manifest.get('complete') or manifest['count_per_arm'] != row['synthetic_questions']:
            raise ValueError('A complete matched synthetic pair is required before training')
        for key in ('case', 'dataset', 'document_sha256', 'prompt_sha256'):
            if manifest[key] != row[key]:
                raise ValueError('Synthetic pair belongs to different evaluation inputs')
        row['data_files'] = manifest['data_files']
        row['data_sha256'] = manifest['data_sha256']
    for key, path in row['data_files'].items():
        if sha_file(path) != row['data_sha256'][key]:
            raise ValueError(f'Immutable reused source changed: {path}')
    specific = read_conversations(row['data_files']['specific_targets'])
    agnostic = read_conversations(row['data_files']['agnostic_targets'])
    count = row['synthetic_questions']
    if len(specific) != count or len(agnostic) < count or count < 1:
        raise ValueError('Existing Teacher data cannot meet the original matched question budget')
    return {'vanilla_cartridges':agnostic[:count], 'contextual_cartridges':specific}


def initial_cache(engine, config, row, document, deadline):
    """Reuse an immutable, document-only initialization, never a trained KV."""
    from contextualized.chunked_initialization import initialize_document
    from contextualized.cache_io import load_cartridge, save_cartridge
    root = Path(config['output'])/'initialization'/row['document_key']
    root.mkdir(parents=True, exist_ok=True)
    prefix = row['prefix_tokens']
    path = root/f'p{prefix}.safetensors'
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return load_cartridge(path, device=engine.device,
                expected_prefix_tokens=prefix, expected_model=config['model'])[0]
        cache = initialize_document(engine, document, prefix,
            chunk_tokens=config['initialization_chunk_tokens'], deadline=deadline)
        save_cartridge(cache, path, config['model'])
        write(path.with_suffix('.json'), dict(document_sha256=row['document_sha256'],
            prefix_tokens=prefix, model=config['model'],
            initialization='upstream_KVFromText_tokenization_chunked_native_forward',
            trained=False, frozen_first_tokens=cache._num_frozen_tokens))
        return cache


def sample_parameters(cache):
    """Small fixed-stride diagnostic; avoids copying a 7+ GB KV to score updates."""
    import torch
    return torch.cat([p.detach().reshape(-1)[::max(1,p.numel()//1024)][:1024].float().cpu()
                      for p in cache.parameters() if p.requires_grad])


def run_pair(engine, config, row, case, deadline):
    import torch
    from contextualized.ed_jsd import fit_ed_jsd
    from contextualized.early_study import make_dataset, evaluate_distillation
    from contextualized.cache_io import save_cartridge, load_cartridge
    from experiments.run_figure4_contextual import complete_target_layout, evaluate_frozen_cartridge
    root = Path(config['output'])
    directory = root/'training'/row['dataset']/row['case']/budget_label(row)
    directory.mkdir(parents=True, exist_ok=True)
    conversations = load_targets(row)
    seed = config['seed'] + row['position']
    configs, packed_counts = {}, []
    for method in METHODS:
        cfg, packing = complete_target_layout(engine, conversations[method], config['training'])
        configs[method] = (cfg, packing)
        engine.dataset_packing_mode = cfg['packing_mode']
        packed_counts.append(len(make_dataset(engine, conversations[method], cfg['packed_tokens'], seed)))
    total = matched_updates(packed_counts, config['training']['gradient_accumulation_steps'],
        config['training']['minimum_epochs'], config['training']['minimum_updates'])
    for method in METHODS:
        destination = result_path(root, row, method)
        if destination.exists():
            continue
        if time.time() >= deadline - 60:
            raise TimeoutError('Allocation deadline before next paired arm')
        cfg, packing = configs[method]
        cfg = dict(cfg, fixed_updates=total, ed_jsd_warmup_steps=total//2,
                   prefix_tokens=row['prefix_tokens'], model=config['model'])
        engine.prefix_tokens = row['prefix_tokens']
        engine.packed_tokens = cfg['packed_tokens']
        engine.dataset_packing_mode = cfg['packing_mode']
        contract = dict(protocol=config['protocol'], config=cfg, seed=seed,
                        source=row, method=method)
        contract_file = directory/f'{method}.contract.json'
        if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
            raise ValueError('Refuse to mix retraining contracts')
        write(contract_file, contract)
        checkpoint, record = directory/f'{method}.safetensors', directory/f'{method}.training.json'
        if checkpoint.exists() and record.exists():
            training = json.loads(record.read_text())
            cache, _ = load_cartridge(checkpoint, device=engine.device,
                expected_prefix_tokens=row['prefix_tokens'], expected_model=config['model'])
        else:
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            cache = initial_cache(engine, config, row, case['document'], deadline)
            if cache.num_cartridge_tokens() != row['prefix_tokens']:
                raise RuntimeError('Wrong actual native KV size')
            before = sample_parameters(cache)
            fixed_data = make_dataset(engine, conversations[method], cfg['packed_tokens'], seed)
            initial_loss = evaluate_distillation(engine, cache, fixed_data)
            def progress(item):
                with (directory/f'{method}.progress.jsonl').open('a') as handle:
                    handle.write(json.dumps(item, allow_nan=False)+'\n')
            progress(dict(event='initial_fixed_training_loss', value=initial_loss,
                          benchmark_reference_used=False))
            training = fit_ed_jsd(engine, cache, conversations[method], cfg, seed,
                selective=method=='contextual_cartridges', deadline=deadline, progress=progress)
            final_loss = evaluate_distillation(engine, cache, fixed_data)
            after = sample_parameters(cache)
            delta = after-before
            changed = int((delta != 0).sum())
            if changed == 0:
                raise RuntimeError('Sampled native KV did not change despite optimization')
            training['learning_diagnostics'] = dict(
                same_fixed_training_data_before_after=True, benchmark_answers_used=False,
                initial_sparse_teacher_CE=initial_loss, final_sparse_teacher_CE=final_loss,
                relative_training_loss_reduction=(initial_loss-final_loss)/max(abs(initial_loss),1e-12),
                sampled_KV_coordinates=len(before), sampled_changed_coordinates=changed,
                sampled_relative_KV_l2_change=float(delta.norm()/before.norm().clamp_min(1e-12)),
                diagnostic_is_accuracy=False)
            training['initialization_training_diagnostics_seconds'] = time.monotonic()-began
            training['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            training['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            # Durable final KV before evaluation; remove only after the result
            # is saved. An interrupted decode need not repeat SD.
            save_cartridge(cache, checkpoint, config['model'])
            write(record, training)
            del before, after, delta, fixed_data
        prediction, metric = evaluate_frozen_cartridge(engine, cache, row['dataset'], case)
        if not math.isfinite(metric):
            raise FloatingPointError('Nonfinite benchmark score')
        physical = sum(p.numel()*p.element_size() for p in cache.parameters())
        denominator = row['whole_kv']['prefix_tokens']
        if 'requested_prefix_tokens' not in row and cache.num_cartridge_tokens()/denominator > .20:
            raise RuntimeError('Native KV exceeded the authorized 20% maximum')
        write(destination, dict(protocol=config['protocol'], method=method, dataset=row['dataset'],
            case=case['id'], model=config['model'], prediction=prediction, reference=case['reference'],
            metric_value=metric, training=training, packing=packing, synthetic_questions=len(conversations[method]),
            reused_source_paths=row['data_files'], reused_source_sha256=row['data_sha256'],
            document_sha256=row['document_sha256'], prompt_sha256=row['prompt_sha256'],
            cache=dict(prefix_tokens=cache.num_cartridge_tokens(), cache_gb=physical/1e9,
                percentage=row.get('percent'), requested_prefix_tokens=row.get('requested_prefix_tokens'),
                actual_ratio=cache.num_cartridge_tokens()/denominator,
                denominator_whole_kv_tokens=denominator,
                original_document_tokens=row['whole_kv']['document_tokens'],
                whole_kv_source_was_truncated=row['whole_kv']['truncated'],
                cache_measure='native_bf16_K_V_bytes', matched_whole_kv_source=row['whole_kv']['path']),
            seed=seed, benchmark_answer_used_for_training=False,
            synthesis_regenerated=bool(row.get('synthesis_manifest')),
            teacher_logits_regenerated=bool(row.get('synthesis_manifest')),
            runtime_code_root=config['runtime_code_root'], evaluation_only_reference=True))
        cache.clear(); del cache
        gc.collect(); torch.cuda.empty_cache()
        if not config['retain_finished_checkpoints']:
            checkpoint.unlink(missing_ok=True)


def worker(config, deadline, rank):
    import torch
    from baselines.worker import unit_log
    from baselines.dispatch import claim
    from experiments.syn_sd_factorial_runtime import load_engine, resident
    import torch._dynamo.config as compile_config
    compile_config.recompile_limit = max(compile_config.recompile_limit, 256)
    output = Path(config['output'])
    plan = json.loads(Path(config['plan']).read_text())
    datasets = {}
    engine = None
    for row in plan:
        if 'worker_rank' in row and row['worker_rank'] != rank:
            continue
        if row.get('synthesis_manifest') and not Path(row['synthesis_manifest']).exists():
            continue  # generation recorded its explicit shortfall; never train a smaller dataset
        if time.time() >= deadline - 60:
            return 75
        failure = output/'failures'/f"{row['index']:05d}.json"
        if complete(output, row) or failure.exists():
            continue
        lock = claim(output/'claims', row['index'])
        if lock is None:
            continue
        try:
            if complete(output, row) or failure.exists():
                continue
            write(output/f'worker-{rank}.json', dict(state='running', unit=row['index'],
                dataset=row['dataset'], case=row['case'], budget=budget_label(row), epoch=time.time()))
            log = output/'logs'/f"pair-{row['index']:05d}.log"
            with unit_log(log):
                try:
                    if engine is None:
                        engine = load_engine(dict(config['training'], model=config['model'], prefix_tokens=512))
                        resident(engine, 'student')
                        engine.sd_target_only_logits = True
                        engine.sd_offload_saved_tensors = True
                        if engine.teacher is not None:
                            raise RuntimeError('A cached-only job must not load a Teacher')
                    free, total = torch.cuda.mem_get_info()
                    if row['memory']['training_peak_estimate_bytes'] > total*.95:
                        raise MemoryError('GPU capacity is below the precomputed training memory allowance')
                    if row['dataset'] not in datasets:
                        datasets[row['dataset']] = {c['id']:c for c in load_cases(row['dataset'], config['dataset_root'])}
                    case = datasets[row['dataset']][row['case']]
                    if (hashlib.sha256(case['document'].encode()).hexdigest() != row['document_sha256']
                            or hashlib.sha256(case['prompt'].encode()).hexdigest() != row['prompt_sha256']):
                        raise ValueError('Evaluation inputs changed after planning')
                    run_pair(engine, config, row, case, deadline)
                except TimeoutError as error:
                    print(str(error), flush=True)
                    write(output/f'worker-{rank}.json', dict(state='deadline', unit=row['index'], epoch=time.time()))
                    return 75
                except Exception as error:
                    traceback.print_exc()
                    write(failure, dict(unit=row, rank=rank, error=str(error), log=str(log), epoch=time.time()))
                    return 76  # clean process restart; this failed pair is not retried
            write(output/f'worker-{rank}.json', dict(state='advancing', completed_unit=row['index'], epoch=time.time()))
        finally:
            lock.close()
    write(output/f'worker-{rank}.json', dict(state='finished', epoch=time.time()))
    return int(bool(list((output/'failures').glob('*.json'))))


def dispatch(config_path):
    from experiments.gpu_campaign_clock import start_execution_window
    from experiments.dispatch_figure4_contextual import runtime_for_gpu
    config = start_execution_window(config_path)
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    if rank not in (0, 1) or config['workers'] != 2:
        raise ValueError('Exactly two one-GPU workers required')
    deadline = config['deadline_epoch']
    if os.environ.get('SLURM_JOB_END_TIME', '').isdigit():
        deadline = min(deadline, float(os.environ['SLURM_JOB_END_TIME']))
    deadline -= 180
    names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True)
    python, runtime = runtime_for_gpu(config, names)
    repo, base = Path(config['runtime_code_root']), Path(config['workspace'])
    cuda = Path('/apps/software/system/software/CUDA/12.8.0')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{repo}:{base}/.deps:{repo}/upstream',
        CARTRIDGES_DIR=str(base/'cartridges'), CARTRIDGES_OUTPUT_DIR=config['output'],
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled', RETRAIN_DEADLINE=str(deadline),
        CUDA_HOME=str(cuda), CUDA_ROOT=str(cuda), CUDA_PATH=str(cuda))
    env['PATH'] = f'{Path(python).parent}:{base}/.deps/toolchain/bin:{cuda}/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH'] = f'{cuda}/targets/x86_64-linux/lib:'+env.get('LD_LIBRARY_PATH','')
    for name, sub in {'TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton',
        'TORCH_EXTENSIONS_DIR':'extensions','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
        path = Path(config['runtime_cache_root'])/runtime/sub
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    child = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps(dict(event='ratio_retrain_dispatch', rank=rank, gpu_names=names.strip(),
        runtime=runtime, deadline=deadline, generation_enabled=False)), flush=True)
    while not stopping and time.time() < deadline-30:
        child = subprocess.Popen([python, '-u', str(repo/'experiments/run_ratio_retrain.py'),
            '--config',config_path,'--worker'], cwd=repo, env=env, start_new_session=True)
        try:
            code = child.wait(timeout=max(1,deadline-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            return 75
        if code != 76:
            return code
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        cfg = json.loads(Path(args.config).read_text())
        raise SystemExit(worker(cfg, float(os.environ['RETRAIN_DEADLINE']), int(os.environ['SLURM_PROCID'])))
    raise SystemExit(dispatch(args.config))
