"""Run SnapKV, official FastKV and summaries under the same native-KV byte caps."""
import gc
import json
import math
from pathlib import Path
import time

from baselines.protocol import result_path, digest, cache_statistics
from baselines.ratio_budget import native_bytes_per_token, biased_slots, smaller_target, validate_storage
from experiments.run_figure4_contextual import write


def budget_summary(runtime, document, slots, directory):
    """Summaries use a capacity-dependent intermediate budget, not a 256-token bottleneck."""
    ids = runtime.ids(document)
    wrapper = len(runtime.ids(runtime.format_document('')))
    allowance = slots-wrapper-4
    if allowance < 1:
        raise ValueError('Summary budget does not fit the fixed chat wrapper')
    path = directory/f'p{slots}.summary.json'
    if path.exists():
        text = json.loads(path.read_text())['text']
    else:
        chunk = runtime.config['summary_chunk_tokens']
        if len(ids) <= chunk:
            material = document
        else:
            parts = []
            for begin in range(0, len(ids), chunk):
                runtime.check_time()
                end = min(begin+chunk, len(ids))
                intermediate_limit = min(end-begin, max(256, math.ceil(2*allowance*(end-begin)/len(ids))))
                part_path = directory/f'p{slots}.part-{begin:07d}.json'
                if part_path.exists():
                    part = json.loads(part_path.read_text())['text']
                else:
                    part = runtime._summary(runtime.tokenizer.decode(ids[begin:end]), intermediate_limit)
                    write(part_path, dict(text=part, original_token_span=[begin,end],
                        generation_budget=intermediate_limit, query_used=False, gold_used=False))
                parts.append(part)
            material = '\n\n'.join(parts)
        # Inputs to the final generation also obey the native window.
        if len(runtime.ids(material))+allowance+2048 > runtime.model.config.max_position_embeddings:
            raise ValueError('Summary generation would exceed the native model window')
        text = runtime._summary(material, allowance)
        encoded = runtime.ids(text)
        while len(runtime.ids(runtime.format_document(text))) > slots:
            excess = len(runtime.ids(runtime.format_document(text)))-slots
            encoded = encoded[:max(0,len(encoded)-excess-1)]
            text = runtime.tokenizer.decode(encoded, skip_special_tokens=False)
            if not encoded:
                raise ValueError('Summary cannot fit the matched memory budget')
        write(path, dict(text=text, original_tokens=len(ids), final_token_budget=allowance,
            formatted_tokens=len(runtime.ids(runtime.format_document(text))),
            query_used=False, gold_used=False, intermediate_policy='twice_final_budget_proportional_to_chunk_length'))
    return text


def fit_fastkv_to_bytes(runtime, document, slots, byte_unit, directory):
    """Preserve the authors' algorithm; adjust target size using memory only.

    Nonuniform head padding and attention biases count toward the byte ceiling.
    Never trim solved K/V tensors or use benchmark scores to choose a cache.
    """
    import torch
    cap = slots*byte_unit
    model = runtime.model.config
    target = biased_slots(slots, getattr(model, 'head_dim', None) or model.hidden_size // model.num_attention_heads)
    attempts = []
    for _ in range(8):
        runtime.check_time()
        compacted, stats = runtime.attention_matching(document, target, include_wrapper_in_budget=True)
        physical = sum(t.numel()*t.element_size() for layer in compacted for t in layer)
        attempts.append(dict(target_total_slots=target, measured_bytes=physical, byte_ceiling=cap))
        if physical <= cap:
            return compacted, stats, attempts
        del compacted
        gc.collect(); torch.cuda.empty_cache()
        target = smaller_target(target, physical, cap)
    raise ValueError('Official FastKV did not fit the matched byte ceiling after size-only retries')


def run_ratio_unit(config, unit, cases, directory, deadline, runtime):
    import torch
    from baselines.protocol import bounded_context_ids
    document, method, dataset = cases[0]['document'], unit['method'], unit['dataset']
    model_cfg = runtime.model.config.to_dict()
    byte_unit = native_bytes_per_token(model_cfg)
    reference = unit['whole_kv']
    seed = config['seed']+int(unit['document_key'][:8],16)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    def emit(case, budget, prediction, score, stats, **extra):
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite metric')
        path = result_path(config['output'],unit,budget['prefix_tokens'],case['id'])
        if path.exists():
            return
        if stats['physical_cache_bytes'] > budget['prefix_tokens']*byte_unit:
            raise ValueError('Attempted to report an over-budget cache')
        write(path, dict(protocol=config['protocol'], method=method, dataset=dataset,
            case=case['id'], model=config['model'], prediction=prediction, reference=case['reference'],
            metric_value=score, document_sha256=digest(document), prompt_sha256=digest(case['prompt']),
            cache=dict(stats, percentage=budget['percent'],
                requested_native_slots=budget['prefix_tokens'],
                denominator_whole_kv_tokens=reference['prefix_tokens'],
                actual_whole_kv_byte_ratio=stats['physical_cache_bytes']/(reference['prefix_tokens']*byte_unit)),
            cartridge_training_hyperparameters_applicable=False,
            budget_definition='same_stored_memory_upper_bound_including_bias_padding_wrapper',
            benchmark_answer_used_for_construction=False,
            runtime_code_root=config['runtime_code_root'], **extra))
    budgets = unit['ratio_budgets']
    if method == 'snapkv':
        source, source_stats = runtime.make_source_cache(document, [c['prompt'] for c in cases])
        length = source_stats['prefix_tokens']
        if length != reference['prefix_tokens']:
            raise ValueError('SnapKV source differs from the measured Whole-KV denominator')
        for case in cases:
            pending = [b for b in budgets if not result_path(config['output'],unit,b['prefix_tokens'],case['id']).exists()]
            if not pending:
                continue
            queries = runtime.snap_queries(source,length,case['prompt'])
            for budget in pending:
                slots = budget['prefix_tokens']
                compacted = runtime.snap_cache(source, queries, biased_slots(slots, getattr(runtime.model.config, 'head_dim', None) or runtime.model.config.hidden_size // runtime.model.config.num_attention_heads))
                stats = cache_statistics(compacted)
                actual = sum(t.numel()*t.element_size() for layer in compacted for t in layer)
                stats.update(validate_storage(actual,slots,byte_unit))
                prediction,score = runtime.evaluate_snap(compacted,queries,length,case,dataset)
                emit(case,budget,prediction,score,stats,source=source_stats,
                     variant='actual-query SnapKV, GQA observation=32, average pooling=5')
                del compacted
                gc.collect(); torch.cuda.empty_cache()
            del queries
        del source
        return
    for budget in budgets:
        slots=budget['prefix_tokens']
        pending=[c for c in cases if not result_path(config['output'],unit,slots,c['id']).exists()]
        if not pending:
            continue
        runtime.check_time()
        if method == 'summarization':
            text = budget_summary(runtime,document,slots,directory)
            source,stats=runtime.make_source_cache(text,[c['prompt'] for c in pending])
            if stats['truncated']:
                raise ValueError('Summary unexpectedly exceeded the evaluation window')
            length=stats['prefix_tokens']
            stats.update(validate_storage(runtime.source_bytes(source),slots,byte_unit))
            for case in pending:
                try:
                    prediction,score=runtime.evaluate(source,length,case,dataset)
                finally:
                    source.crop(length)
                emit(case,budget,prediction,score,stats,source=dict(original_document_tokens=len(runtime.ids(document))),
                    query_used_in_summary=False,variant='hierarchical budget-aware query-agnostic summary')
            del source
        elif method == 'fast_kv':
            path,metadata=directory/f'p{slots}.cache.pt',directory/f'p{slots}.cache.json'
            if path.exists() and metadata.exists():
                compacted=torch.load(path,map_location=runtime.device,weights_only=True)
                stats=json.loads(metadata.read_text())
            else:
                # Retain the source policy of the official text-chunked method:
                # all document chunks are available, unlike window-limited ICL.
                compacted,raw,attempts=fit_fastkv_to_bytes(runtime,document,slots,byte_unit,directory)
                stats={k:raw[k] for k in ('variant','num_chunks','target_ratio',
                    'original_sequence_length','total_original_article_tokens',
                    'effective_compacted_seq_len','tensor_compacted_seq_len')}
                stats.update(cache_statistics(compacted))
                actual=sum(t.numel()*t.element_size() for layer in compacted for t in layer)
                stats.update(validate_storage(actual,slots,byte_unit),size_only_budget_attempts=attempts)
                stats['logical_positions_exceed_native_window']=(
                    stats['original_sequence_length'] > runtime.model.config.max_position_embeddings)
                temporary=path.with_suffix('.tmp')
                torch.save(tuple(tuple(t.cpu() for t in layer) for layer in compacted),temporary)
                temporary.replace(path)
                write(metadata,stats)
            for case in pending:
                runtime.check_time()
                prediction,score=runtime.evaluate_compacted(compacted,stats['original_sequence_length'],case,dataset)
                emit(case,budget,prediction,score,stats)
            del compacted
        else:
            raise ValueError('Unsupported ratio baseline')
        gc.collect(); torch.cuda.empty_cache()
