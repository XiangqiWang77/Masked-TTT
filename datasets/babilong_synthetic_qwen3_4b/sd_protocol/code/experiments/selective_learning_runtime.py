"""Original source-supported admission, applied to native Cartridges SD.

The admission chain follows Stack_TTT's
TaskConditionedSupportSelfDistillation: focused/full grounded-veto critics,
candidate-blind extraction, source quote checks, generic verification, and a
fresh task-conditioned verification. The last verification must explicitly
establish support. Its uncertainty fallback is not an admission permission.

Only the old admission logic is reused. Its memory parameterization and
optimizer are not imported into training. Returned conversations carry the
unchanged evidence-conditioned Teacher's real sparse token distributions;
the caller trains them with the existing Cartridges SD trainer.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time


SYSTEM_TEMPLATE = "\nYou are in a conversation about the following user information.\n\n<info>\n{subcorpus}\n</info>"


def _legacy_helpers(stack_ttt_root):
    """Import the original prompts/parsers rather than reinterpret the gate."""
    path = str(Path(stack_ttt_root).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return (
        importlib.import_module('stack_ttt.independent_evidence_judge'),
        importlib.import_module('stack_ttt.teacher_update_judge'),
    )


class FrozenAssessmentGenerator:
    """Greedy, single-user-message generation, matching the old critic format.

    Every assessment starts from source text with an empty cache. It never
    accesses or changes the trainable Cartridge, and never generates targets
    for the Student. Chunked prefill bounds transient attention allocations.
    """
    def __init__(self, engine, *, prefill_tokens=1024):
        if prefill_tokens < 1:
            raise ValueError('Positive critic prefill size required')
        self.engine = engine
        self.prefill_tokens = prefill_tokens

    def __call__(self, content, max_new_tokens):
        import torch
        engine = self.engine
        ids = engine.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        maximum = getattr(engine.teacher.config, 'max_position_embeddings', None)
        if maximum is not None and len(ids) + max_new_tokens > maximum:
            raise ValueError('Source critic exceeds the model context limit')
        generated = []
        past = None
        with torch.no_grad():
            for start in range(0, len(ids), self.prefill_tokens):
                tokens = torch.tensor([ids[start:start + self.prefill_tokens]],
                                      device=engine.device, dtype=torch.long)
                output = engine.teacher(input_ids=tokens, past_key_values=past,
                                        use_cache=True, logits_to_keep=1)
                past = output.past_key_values
            for step in range(max_new_tokens):
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                if token in engine.eos_ids:
                    break
                if step + 1 < max_new_tokens:
                    output = engine.teacher(
                        input_ids=torch.tensor([[token]], device=engine.device),
                        past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = output.past_key_values
        complete = bool(generated and generated[-1] in engine.eos_ids)
        return {'prediction': engine.tokenizer.decode(generated, skip_special_tokens=True),
                'generated_token_ids': generated, 'prompt_tokens': len(ids),
                'complete': complete, 'finish_reason': 'eos' if complete else 'length'}


def assess_teacher_signal(*, document, question, task_prompt, evidence, candidate,
                          candidate_complete, generate_assessment, legacy_helpers,
                          evidence_options=None, judge_options=None,
                          admission_policy='task_conditioned_support_required'):
    """Execute the old strict-support gate without its old optimizer/cache API.

    ``generate_assessment(content, max_new_tokens)`` returns the original
    prediction/complete fields. Candidate text is never rewritten. Extraction
    itself is candidate-blind; as in the old implementation, source quotes
    from earlier candidate-aware critics are subsequently added to its quotes.
    """
    independent, original = legacy_helpers
    evidence_options = evidence_options or independent.EvidenceJudgeOptions()
    judge_options = judge_options or original.SelfJudgeOptions(
        max_new_tokens=384, focused_max_new_tokens=256,
        policy='grounded_veto', focused_hint_policy='verified_quotes_only')
    if (judge_options.policy != 'grounded_veto'
            or judge_options.focused_hint_policy != 'verified_quotes_only'):
        raise ValueError('The old strict gate requires verified-quotes grounded veto')
    if not document or task_prompt.count(document) != 1:
        raise ValueError('Require the complete original document exactly once in the task')
    if not question.strip() or question not in task_prompt or not evidence.strip():
        raise ValueError('Require the actual learning question and selected source text')

    focused_prompt = 'Selected source passages, in their original order:\n' + evidence
    focused_generation = generate_assessment(original.judge_user_content(
        focused_prompt, candidate, question=question, policy='grounded_veto'),
        judge_options.focused_max_new_tokens)
    focused = dict(focused_generation, **original.parse_grounded_veto(
        focused_generation['prediction'], evidence,
        complete=focused_generation['complete'], candidate_complete=candidate_complete))
    hint = original.focused_source_hint(focused, evidence, document, 'verified_quotes_only')
    focused.update(context_view='fresh_selected_source_only',
                   full_judge_hint_policy='verified_quotes_only',
                   hint_supplied_to_full_judge=hint)

    full_generation = generate_assessment(original.judge_user_content(
        task_prompt, candidate, question=question, policy='grounded_veto',
        focused_assessment=hint), judge_options.max_new_tokens)
    old_judgment = dict(full_generation, **original.parse_grounded_veto(
        full_generation['prediction'], document, complete=full_generation['complete'],
        candidate_complete=candidate_complete))
    old_judgment.update(focused_assessment=focused, context_view='full_source_and_original_task')

    if admission_policy=='verified_quote_grounded_veto':
        # An unresolved assessment cannot remove ordinary self-study. Token-
        # limited continuations remain eligible, as in the original SD data.
        contradicted=bool(old_judgment.get('grounded_contradiction'))
        no_answer=bool(old_judgment.get('parse_valid') and
                       old_judgment.get('declared_candidate_has_answer') is False)
        approved=not contradicted
        return {'apply_update':approved,'verdict':'APPLY' if approved else 'SKIP',
            'reason':old_judgment['reason'],
            'decision_source':'verified_quote_grounded_veto',
            'decision_rule':'retain original SD unless source-grounded contradiction vetoes it; uncertainty and missing direct answers do not veto',
            'original_pre_update_judge':old_judgment,
            'task_assessment':old_judgment,'uncertainty_rejects_signal':False,
            'partial_official_teacher_prefix_remains_eligible':True,
            'candidate_complete':candidate_complete,'grounded_contradiction':contradicted,
            'no_answer_detected_but_not_a_veto':no_answer,'teacher_rewritten':False,
            'reference_answers_used':False,'executed_before_optimizer':True,
            'optimizer_steps_before_judge':0}
    if admission_policy not in {'task_conditioned_support_required','original_task_conditioned_support_required'}:
        raise ValueError('Unknown evidence-supported learning admission policy')

    extraction_content = (task_prompt + '\n\n' + independent.EXTRACTION
        + '\nOriginal question (JSON):\n' + json.dumps(question, ensure_ascii=False))
    extracted = generate_assessment(extraction_content, evidence_options.extraction_tokens)
    data = independent.parse_object(extracted['prediction']) if extracted['complete'] else None
    valid = (isinstance(data, dict) and list(data) == ['evidence']
        and isinstance(data['evidence'], list)
        and len(data['evidence']) <= evidence_options.maximum_quotes
        and all(isinstance(q, str) and q.strip() for q in data['evidence']))
    quotes = list(data['evidence']) if valid else []
    # Preserve the original chain's exact quote augmentation, not old verdicts.
    for assessment in (old_judgment, focused):
        quotes.extend(assessment.get('evidence', []))
    occurrences, ambiguity = independent.verified_occurrences(
        quotes, document, evidence_options.maximum_occurrences)
    empty = dict(resolved=False, apply_update=None, parse_valid=False,
        reason=ambiguity or 'no_source_verified_evidence', evidence=[],
        source_quotes_verified=False)
    generic = dict(empty)
    generic_generation = None
    task_assessment = dict(empty)
    task_generation = None
    if occurrences:
        suffix = ('\n\n' + independent.VERIFICATION
            + '\nOriginal question (JSON):\n' + json.dumps(question, ensure_ascii=False)
            + '\nUnchanged candidate (JSON):\n' + json.dumps(candidate, ensure_ascii=False))
        source_view = ('Verbatim source excerpts, in original document order:\n'
                       + json.dumps(occurrences, ensure_ascii=False))
        generic_generation = generate_assessment(
            source_view + suffix, evidence_options.verification_tokens)
        generic = independent.parse_verification(generic_generation['prediction'], document,
                                                  complete=generic_generation['complete'])
        # The task check reuses the same verified occurrences, including all
        # repeated occurrences, exactly as task_conditioned_evidence_judge.
        verified, changed = independent.verified_occurrences(
            [item['quote'] for item in occurrences], document,
            evidence_options.maximum_occurrences)
        if changed or verified != occurrences:
            raise ValueError('The fixed source occurrences changed or lack source support')
        start = task_prompt.index(document)
        task = (task_prompt[:start]
            + '[Original source excerpts for this evidence check appear below.]'
            + task_prompt[start + len(document):])
        task_content = ('Original task instructions and question:\n' + task
                        + '\n\n' + source_view + suffix)
        task_generation = generate_assessment(task_content, evidence_options.verification_tokens)
        task_assessment = independent.parse_verification(task_generation['prediction'], document,
                                                        complete=task_generation['complete'])
    approved = bool(task_assessment['resolved'] and task_assessment['apply_update'])
    return {
        'apply_update': approved, 'verdict': 'APPLY' if approved else 'SKIP',
        'reason': task_assessment['reason'],
        'decision_source': 'task_conditioned_support_required',
        'decision_rule': 'apply only when task-aware source verification establishes support',
        'original_pre_update_judge': old_judgment,
        'extraction': extracted, 'extraction_parse_valid': valid,
        'independent_extraction_had_no_candidate': True,
        'earlier_critics_source_quotes_appended': True,
        'earlier_critics_were_candidate_aware': True,
        'verified_source_occurrences': occurrences,
        'parent_generic_assessment': generic, 'parent_generic_generation': generic_generation,
        'parent_generic_verification_executed_but_not_used_for_gate': True,
        'task_assessment': task_assessment, 'task_verification': task_generation,
        'verification_context_view': 'original_task_plus_fixed_source_excerpts',
        'uncertainty_rejects_signal': True, 'old_gate_fallback_used': False,
        'old_verdict_or_reason_supplied_to_verification': False,
        'reference_answers_used': False, 'teacher_rewritten': False,
        'executed_before_optimizer': True, 'optimizer_steps_before_judge': 0,
    }


def _conversation(engine, question, evidence, generated, number, original_question):
    import numpy as np
    from cartridges.clients.base import TopLogprobs
    from cartridges.structs import Conversation
    targets = TopLogprobs(
        token_ids=generated.teacher_ids.detach().cpu().numpy().astype(np.int64),
        logprobs=generated.teacher_logprobs.detach().cpu().numpy().astype(np.float32)).flatten(.99)
    return Conversation(messages=[
        Conversation.Message(role='user', content=question,
            token_ids=engine.tokenizer.encode(question, add_special_tokens=False)),
        Conversation.Message(role='assistant',
            content=engine.tokenizer.decode(generated.answer_ids, skip_special_tokens=True),
            token_ids=list(generated.answer_ids), top_logprobs=targets)],
        system_prompt=SYSTEM_TEMPLATE.format(subcorpus=evidence),
        metadata={'question_index': number, 'original_question': original_question,
                  'supervision': 'unchanged_evidence_teacher_sparse_SD',
                  'raw_teacher_topk': 20, 'benchmark_answers_used': False},
        type='query_guided_standard_SD')


async def prepare_selective_conversations(engine, cache, questionrows, sourcechunks, *, document,
        seed=42, answer_tokens=128, concise_answers=False, deadline_epoch=None,
        stack_ttt_root='/home/da839/Stack_TTT', task_prompt_builder=None,
        generate_assessment=None, evidence_options=None, judge_options=None):
    """Return ordinary query SD and its source-supported subset, before fitting.

    Synthetic answer fields are never used as targets or verdicts. A fresh
    evidence-conditioned Teacher provides answer token IDs and soft targets.
    A rejected signal remains in the ordinary query-SD control arm only.
    No replacement target, refusal, or fallback update is fabricated.

    ``task_prompt_builder(document, question)`` may preserve task-specific
    instructions. Its output must contain the entire source exactly once and
    the current synthetic question. ``cache`` is accepted for runner API
    compatibility and is never read or modified during this preparation.
    """
    import torch
    legacy = _legacy_helpers(stack_ttt_root)
    generate_assessment = generate_assessment or FrozenAssessmentGenerator(engine)
    sources = {}
    for row in sourcechunks:
        index = int(row['source_index'])
        begin, end = map(int, row['span'])
        if (index in sources or not 0 <= begin < end <= len(document)
                or document[begin:end] != row['text']):
            raise ValueError('Require unique exact original source spans')
        sources[index] = row
    if task_prompt_builder is None:
        task_prompt_builder = lambda source, question: (
            SYSTEM_TEMPLATE.format(subcorpus=source) + '\n\n' + question)
    deadline = (float(deadline_epoch) if deadline_epoch is not None
                else float(os.environ.get('PAIR_DEADLINE_EPOCH', 'inf')))
    ordinary, selective, records = [], [], []
    began = time.monotonic()
    deadline_reached = False
    for number, row in enumerate(questionrows):
        if time.time() >= deadline:
            deadline_reached = True
            break
        started = time.monotonic()
        original_question = row['question']
        question = original_question
        if concise_answers:
            question += ('\n\nAnswer with only a short phrase. Do not include explanations, '
                         'quotations, or introductory text.')
        indices = sorted({int(c['source_index']) for c in row['evidence']['source_chunks']},
                         key=lambda i: sources[i]['span'][0])
        if not original_question.strip() or not indices:
            records.append({'question_index': number, 'original_question': original_question,
                            'status': 'invalid_question_or_empty_evidence'})
            continue
        evidence = '\n\n'.join(sources[i]['text'] for i in indices)
        print(json.dumps({'event': 'selective_question_started', 'question_index': number,
                          'questions_requested': len(questionrows)}), flush=True)
        torch.manual_seed(seed + number)
        generated = engine.generate_teacher(question, evidence, max_new_tokens=answer_tokens,
                                            sampling_temperature=.6, topk=20)
        if not generated.answer_ids:
            records.append({'question_index': number, 'original_question': original_question,
                            'status': 'empty_teacher_candidate'})
            continue
        conversation = _conversation(engine, question, evidence, generated, number, original_question)
        ordinary.append(conversation)
        candidate = engine.tokenizer.decode(generated.answer_ids, skip_special_tokens=True)
        candidate_ids = list(generated.answer_ids)
        candidate_complete = generated.finished
        del generated
        assessment = assess_teacher_signal(document=document, question=question,
            task_prompt=task_prompt_builder(document, question), evidence=evidence,
            candidate=candidate, candidate_complete=candidate_complete,
            generate_assessment=generate_assessment, legacy_helpers=legacy,
            evidence_options=evidence_options, judge_options=judge_options)
        if assessment['apply_update']:
            accepted = deepcopy(conversation)
            accepted.type = 'query_guided_source_supported_SD'
            accepted.metadata.update(signal_admission='task_conditioned_support_required',
                                     teacher_targets_unchanged=True)
            selective.append(accepted)
        records.append({'question_index': number, 'original_question': original_question,
            'question': question, 'evidence_source_indices': indices,
            'candidate': candidate, 'candidate_token_ids': candidate_ids,
            'candidate_complete': candidate_complete, 'assessment': assessment,
            'status': 'accepted' if assessment['apply_update'] else 'filtered_synthetic_signal',
            'query_sd_conversation_retained': True, 'seconds': time.monotonic() - started})
        print(json.dumps({'event': 'selective_question_complete', 'question_index': number,
            'accepted': assessment['apply_update'], 'seconds': time.monotonic() - started,
            'query_conversations': len(ordinary), 'selective_conversations': len(selective)}), flush=True)
    audit = {'method': 'original_task_conditioned_support_required_gate_on_native_KV_SD',
        'legacy_admission_source': 'stack_ttt.task_conditioned_support_self_distillation',
        'questions_requested': len(questionrows), 'questions_processed': len(records),
        'unprocessed_questions': len(questionrows) - len(records),
        'query_sd_conversations': len(ordinary), 'selective_conversations': len(selective),
        'filtered_questions': len(records) - len(selective), 'questions': records,
        'deadline_reached': deadline_reached, 'seconds': time.monotonic() - began,
        'answer_token_limit': answer_tokens, 'concise_answers': concise_answers,
        'all_signals_filtered': not bool(selective), 'fallback_supervision_used': False,
        'student_updates_in_preparation': 0, 'cartridge_read_or_modified_during_gate': False,
        'critic_context': 'fresh_source_text', 'critic_backbone': 'frozen_teacher',
        'teacher_targets': 'real_top20_raw_logprobs_on_unchanged_generated_answer_prefixes',
        'benchmark_answers_used': False, 'synthetic_answer_labels_used': False,
        'evidence_verification_guarantees_correctness': False}
    return {'query_conversations': ordinary, 'selective_conversations': selective, 'audit': audit}
