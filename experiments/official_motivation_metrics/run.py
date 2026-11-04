#!/usr/bin/env python3
"""Rerun OLMo QC and measure qTTT-style target-attention mass after adaptation."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from types import SimpleNamespace

import torch

from experiments.official_motivation_metrics.attention_mass import AttentionCapture

WORKSPACE = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102/query_cartridges_20261103')
PROTOCOL = WORKSPACE/'runs/olmo3-7b-babilong/protocol.json'
REFERENCES = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/babilong_dense_evidence_20261029/v3/evaluation_references.json')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def gold_positions(engine, request, layout, spans):
    """Map unique, post-hoc gold source character spans to full prompt token indices."""
    text = engine.tokenizer.apply_chat_template(
        [dict(role='user', content=request.prompt)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) + request.answer_prefix
    begin = text.find(request.document)
    if begin < 0 or text.find(request.document, begin+1) >= 0:
        raise ValueError('Cannot uniquely map original document in the prompt')
    enc = engine.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc['input_ids']) != list(layout.ids):
        raise ValueError('Tokenization does not match the evaluated layout')
    targets = [i for i, (a, b) in enumerate(enc['offset_mapping'])
               if layout.initial <= i < layout.document_end and a < b
               and any(a < begin+right and b > begin+left for left, right in spans)]
    if not targets:
        raise ValueError('No annotated support token was mapped')
    return targets


def selected_position_map(layout, plan, separator_length):
    mapping, cursor, previous = {}, layout.initial, layout.initial-1
    for original in plan.original_document_positions:
        if original != previous+1:
            cursor += separator_length
        mapping[original] = cursor
        cursor += 1
        previous = original
    if cursor != plan.document_end:
        raise ValueError('Selected-cache token map does not match its actual document end')
    return mapping


@torch.no_grad()
def measure(engine, reader, branch, suffix, document_end, targets, reference_ids):
    from masked_ttt.runtime import private_forward
    if not reference_ids or suffix.ndim != 1 or not len(suffix):
        raise ValueError('A nonempty common answer prefix and question suffix are required')
    with AttentionCapture(reader.model, targets) as capture:
        output = private_forward(reader, suffix, document_end, branch)
        first_token = int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())
        for i, token in enumerate(reference_ids[:-1]):
            output = private_forward(reader, torch.tensor([token], device=engine.device),
                                     document_end+len(suffix)+i, branch)
    result = capture.result(len(reference_ids))
    if not 0 <= result['attention_mass'] <= 1:
        raise ValueError('Measured attention mass is not a probability')
    return result, first_token


def native_reader(engine):
    from masked_ttt.architectures import install_hybrid_attention
    from masked_ttt.runtime import private_module_tree
    model = private_module_tree(engine.model.model)
    install_hybrid_attention(model)
    return SimpleNamespace(model=model, lm_head=engine.model.lm_head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Run only inside a one-GPU Slurm allocation')
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('OPENAI_API_KEY_FILE', None)
    from masked_ttt import Backbone, FrozenJSONSelector, MaskedTTT, Request, TTTConfig
    from masked_ttt._layout import selected_read_plan
    from masked_ttt.cache import fork_prefix
    from masked_ttt.decision_api import DecisionClient
    from masked_ttt.standalone import build_standalone
    import masked_ttt.pipeline as pipeline

    def reject_network(*_args, **_kwargs):
        raise AssertionError('This diagnostic must use precomputed Decision scores')
    DecisionClient.__init__ = reject_network
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    spec = json.loads(PROTOCOL.read_text())
    plan = json.loads(args.plan.read_text())
    if len(plan['cases']) != plan['case_count'] or not plan['only_complete_original_documents']:
        raise ValueError('Diagnostic plan is inconsistent')
    refs = json.loads(REFERENCES.read_text())
    config = TTTConfig(max_context=spec['native_context'],
        context_reserve_tokens=spec.get('context_reserve_tokens', 4096), steps=2,
        max_answer_steps=8, student_rollout_tokens=8)
    engine = Backbone.load(spec['model_path'], config, local_files_only=True)
    selector = FrozenJSONSelector(spec['shared_scores'])
    method = MaskedTTT(engine, selector)
    source = Path(plan['source_cases'])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    state = {}
    original_prefill, original_standalone = pipeline.prefill_document, pipeline.answer_standalone

    def capture_full(*a, **kw):
        cache = original_prefill(*a, **kw)
        state['full_cache'] = cache
        return cache

    def observe_standalone(local_engine, layout, selected, teacher, trained, limit):
        # Preserve the established run's generation exactly. Gold references are
        # read solely below, after all self-distillation and self-judging.
        output_record = original_standalone(local_engine, layout, selected, teacher, trained, limit)
        current, request = state['current'], state['request']
        answers = refs[current['key']]['answers']
        if not answers or not isinstance(answers[0], str) or not answers[0]:
            raise ValueError('Reference answer unavailable for post-hoc attention measurement')
        answer_ids = local_engine.tokenizer.encode(answers[0], add_special_tokens=False)
        if not answer_ids:
            raise ValueError('Reference answer has no evaluable tokens')
        full_targets = gold_positions(local_engine, request, layout, current['support_character_spans'])
        separator = local_engine.tokenizer.encode('\n\n', add_special_tokens=False)
        plan_selected = selected_read_plan(layout.ids, layout.chunks, layout.initial, selected, separator)
        if (plan_selected.document_end != teacher.document_end or
            tuple(plan_selected.ids[:plan_selected.document_end]) != teacher.document_ids):
            raise ValueError('Teacher document differs from selected diagnostic document')
        mapping = selected_position_map(layout, plan_selected, len(separator))
        compact_targets = [mapping[i] for i in full_targets if i in mapping]
        full_suffix = torch.tensor(layout.ids[layout.document_end:], device=local_engine.device)
        selected_suffix = torch.tensor(plan_selected.ids[plan_selected.document_end:], device=local_engine.device)
        if full_suffix.tolist() != selected_suffix.tolist():
            raise ValueError('Full and selected diagnostic suffixes must be identical')

        native = native_reader(local_engine)
        full, _ = measure(local_engine, native, fork_prefix(state['full_cache'], layout.document_end),
                          full_suffix, layout.document_end, full_targets, answer_ids)
        del native
        native = native_reader(local_engine)
        icl, _ = measure(local_engine, native, fork_prefix(teacher.document_cache, teacher.document_end),
                         selected_suffix, teacher.document_end, compact_targets, answer_ids)
        del native
        qc_reader, qc_branch, qc_suffix, qc_end, _ = build_standalone(
            local_engine, layout, selected, teacher, trained)
        if qc_suffix.tolist() != selected_suffix.tolist():
            raise ValueError('QC and selected ICL question suffix differ')
        qc, qc_first = measure(local_engine, qc_reader.reader, qc_branch, qc_suffix, qc_end,
                               compact_targets, answer_ids)
        if qc_first != output_record['generated_token_ids'][0]:
            raise AssertionError('Attention probing changed the QC first-token prediction')
        state['measurement'] = dict(whole=full, icl=icl, qc=qc,
            common_reference_answer_steps=len(answer_ids),
            full_support_token_count=len(full_targets),
            selected_support_token_count=len(compact_targets),
            metric='mean across all heads, layers and teacher-forced reference answer steps of attention to annotated support tokens',
            reference_answers_used_for_training=False,
            matched_original_question_suffix=True,
            qc_first_token_matches_uninstrumented_generation=True)
        return output_record

    pipeline.prefill_document = capture_full
    pipeline.answer_standalone = observe_standalone
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    save(output/'startup.json', dict(job=os.environ['SLURM_JOB_ID'], model=spec['model_id'],
        gpu=torch.cuda.get_device_name(0), planned_cases=len(plan['cases']),
        online_api_calls=0, single_gpu=True, original_qc_method=True,
        attention_probe='post-training only'))
    consecutive_errors = 0
    newly_completed = 0
    failed_fast = False
    with source.open('rb') as stream:
        for case in plan['cases']:
            if stopping:
                break
            path = output/'cases'/f"{case['key']}.json"
            if path.exists():
                continue
            try:
                stream.seek(case['offset'])
                payload = json.loads(stream.read(case['bytes']))
                if payload['key'] != case['key']:
                    raise ValueError('Input case identity changed')
                request = Request(document=payload['document'], question=payload['query'],
                    prompt=payload['prompt'], answer_prefix=payload.get('answer_prefix', ''),
                    max_new_tokens=payload['max_new_tokens'])
                selector.bind_case(case['key'])
                state.clear()
                state.update(current=case, request=request)
                result = method.answer(request, time_budget_seconds=1800)
                if not result['selection']['complete_original_document']:
                    raise ValueError('Pilot requires a complete original working document')
                if 'measurement' not in state:
                    raise ValueError('Post-training attention measurement was not reached')
                record = dict(key=case['key'], source_question=case['source_question'],
                    task=case['task'], context_tokens=case['context_tokens'],
                    selected_fact_coverage=case['selected_fact_coverage'],
                    complete_selected_evidence=case['selected_fact_coverage'] == 1,
                    attention=state['measurement'], training_steps=result['training']['accepted_steps'],
                    selected_chunks=result['selection']['selected_chunks'],
                    online_api_calls=0, backbone_unchanged=result['backbone_unchanged'])
                save(path, record)
                print(json.dumps(dict(completed=case['key'], attention={k: record['attention'][k]['attention_mass']
                                  for k in ('whole', 'icl', 'qc')})), flush=True)
                consecutive_errors = 0
                newly_completed += 1
                if newly_completed % 4 == 0:
                    subprocess.run([sys.executable, str(Path(__file__).with_name('plot.py')),
                                    '--plan', str(args.plan), '--output', str(output)], check=False)
            except Exception as exc:
                consecutive_errors += 1
                save(output/'errors'/f"{case['key']}.json", dict(key=case['key'],
                    type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
                print(json.dumps(dict(error=case['key'], type=type(exc).__name__, message=str(exc))), flush=True)
                if consecutive_errors >= 2:
                    print('Stopping after two consecutive failed diagnostic cases', flush=True)
                    stopping = True
                    failed_fast = True
            finally:
                state.clear()
                gc.collect()
                torch.cuda.empty_cache()
    print(json.dumps(dict(finished_allocation=True, stopping=stopping)), flush=True)
    if failed_fast:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
