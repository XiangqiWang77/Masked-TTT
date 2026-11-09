"""Paired ordinary/selective SD on the explicitly configured synthetic dataset."""
import argparse
import fcntl
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time
import traceback
import zlib


class TeacherPreparationError(RuntimeError):
    """A failed inference process must stop its worker instead of excluding more cases."""


def use_selective_checkpoint(candidate_loss, reference_loss, *, tolerance,
                             relative_tolerance, has_accepted_signal):
    """Return whether a genuinely selective candidate clears its KD envelope."""
    required_delta = max(float(tolerance),
                         abs(float(reference_loss)) * float(relative_tolerance))
    return (bool(has_accepted_signal)
            and float(candidate_loss) < float(reference_loss) - required_delta), required_delta


def training_views(targets, selected, dataset, sd, config):
    """Choose train/validation signals, including an exact no-signal fallback."""
    selective_consolidation = (
        sd == 'selective_sd' and config.get('selective_consolidation_schedule')
    )
    values = (
        targets[dataset]['train']
        if sd in {'oracle_sd', 'vanilla_sd'} or selective_consolidation
        else selected[dataset]['train']
    )
    consolidation = selected[dataset]['train'] if selective_consolidation else None
    shared_validation = (
        sd == 'selective_sd'
        and config.get('selective_validation_policy') == 'shared_unfiltered_teacher_targets'
    )
    dynamics_without_heldout = (
        sd == 'selective_sd'
        and config.get('selective_validation_policy') ==
            'no_heldout_tracin_training_dynamics'
    )
    heldout = ([] if dynamics_without_heldout else
        targets[dataset]['validation']
        if sd in {'oracle_sd', 'vanilla_sd'} or shared_validation
        else selected[dataset]['validation'])
    fallback = False
    validation_fallback = False
    if not values and config.get('empty_selection_policy') == 'original_SD':
        values = targets[dataset]['train']
        fallback = True
    if (selective_consolidation and not consolidation
            and config.get('empty_selection_policy') == 'original_SD'):
        consolidation = None
        fallback = True
    if fallback:
        # No selected training signal means the entire arm, including its
        # checkpoint objective, must be byte-for-byte ordinary SD semantics.
        heldout = targets[dataset]['validation']
        validation_fallback = True
    elif (not heldout and not dynamics_without_heldout
          and config.get('empty_selection_policy') == 'original_SD'):
        heldout = targets[dataset]['validation']
        validation_fallback = True
    return {
        'values': values,
        'consolidation': consolidation,
        'heldout': heldout,
        'fallback': fallback,
        'validation_fallback': validation_fallback,
        'selective_consolidation': selective_consolidation,
    }


def arm_training_config(config, results, name):
    """Resolve a per-arm budget without changing its distillation objective.

    In packed-update-matched protocols the released Vanilla-SD arm establishes
    the number of optimizer steps.  Every other arm executes that same number
    of one-packed-sequence updates.  This matters for concise query-specific
    answers: two nominal epochs can otherwise mean only two updates, while the
    long released conversations receive many more updates.
    """
    arm_config = dict(config)
    reference = config.get('matched_update_reference_arm')
    if not reference or reference == name:
        return arm_config
    if reference not in results:
        raise ValueError('Matched-update reference arm must finish first')
    updates = int(results[reference]['training']['updates'])
    if updates < 1:
        raise ValueError('Matched-update reference executed no optimizer step')
    arm_config['fixed_updates'] = updates
    if config.get('matched_update_unit') == 'packed_optimizer_steps':
        # ``official_epochs`` derives update count from each arm's own packing,
        # which is precisely the confound this protocol removes.
        arm_config.pop('official_epochs', None)
        arm_config.update(
            train_batch_size=1,
            min_steps=1,
            max_steps=updates,
            eval_every=max(1, updates // 2),
            patience=updates + 1,
            track_best_before_min=True,
        )
        if (name.endswith('_selective_sd')
                and config.get('selective_consolidation_schedule')
                and config.get('selective_consolidation_start_policy') ==
                    'matched_halfway'):
            # Preserve an ordinary-SD first half exactly; evidence selection
            # is only allowed to change the second half of the matched budget.
            arm_config['selective_consolidation_start_step'] = updates // 2 + 1
            if config.get('selective_checkpoint_after_consolidation_only'):
                # The outer envelope already compares the completed selective
                # candidate with the exact Vanilla checkpoint.  Letting the
                # inner trainer restore a warmup-only checkpoint makes the
                # candidate contain no selective update at all, so it cannot
                # measure Selective SD.  Checkpoint the candidate only after
                # the consolidation phase has actually run; the envelope
                # remains the non-harm fallback.
                arm_config['min_steps'] = updates // 2 + 1
                arm_config['track_best_before_min'] = False
    return arm_config


def saved_cartridge_heldout_loss(engine, cartridge, conversations, config, seed):
    """Evaluate a saved arm on exactly the candidate's held-out trajectories."""
    if not conversations:
        return None
    from contextualized.cache_io import load_cartridge
    from contextualized.early_study import make_dataset, evaluate_distillation
    cache, _ = load_cartridge(cartridge, device=engine.device)
    try:
        dataset = make_dataset(engine, conversations, config['packed_tokens'], seed + 1)
        return evaluate_distillation(engine, cache, dataset)
    finally:
        cache.clear()


def tracin_selective_view(engine, initial, vanilla_cartridge,
                          reference_conversations, candidate_conversations,
                          config, seed):
    """Select positive-influence SD signals at initial and Vanilla checkpoints."""
    from contextualized.cache_io import load_cartridge
    from contextualized.tracin_sd import aggregate_checkpoints, score_checkpoint

    threshold = float(config.get('tracin_threshold', 0.0))
    learning_rate = float(config['learning_rate'])
    packed_tokens = int(config['packed_tokens'])
    checkpoints = []
    metadata = []

    rows, meta = score_checkpoint(
        engine, initial, reference_conversations, candidate_conversations,
        packed_tokens=packed_tokens, seed=seed + 7000,
        learning_rate=learning_rate)
    checkpoints.append(rows)
    metadata.append(dict(meta, checkpoint='initial_cartridge'))

    vanilla, _ = load_cartridge(vanilla_cartridge, device=engine.device)
    try:
        rows, meta = score_checkpoint(
            engine, vanilla, reference_conversations, candidate_conversations,
            packed_tokens=packed_tokens, seed=seed + 8000,
            learning_rate=learning_rate)
        checkpoints.append(rows)
        metadata.append(dict(meta, checkpoint='vanilla_SD'))
    finally:
        vanilla.clear()

    decisions = aggregate_checkpoints(checkpoints, threshold=threshold)
    kept = [conversation for conversation, decision in zip(
        candidate_conversations, decisions) if decision['keep']]
    return kept, {
        'method': 'TracIn_positive_influence_selective_SD',
        'reference_objective': 'all_ordinary_SD_training_trajectories',
        'candidate_pool': 'source_admitted_Teacher_trajectories',
        'checkpoints': metadata,
        'threshold': threshold,
        'strict_rule': 'keep iff summed TracIn score > threshold',
        'candidates': len(candidate_conversations),
        'kept': len(kept),
        'deleted': len(candidate_conversations) - len(kept),
        'decisions': decisions,
        'heldout_set_used': False,
        'benchmark_answers_used': False,
        'backbone_frozen': True,
        'native_KV_gradients_only': True,
        'first_order_local_influence_not_correctness_proof': True,
    }




def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def input_rows(item, config, seed):
    """Use the frozen synthetic conversations and their original source bindings.

    Released Cartridges studies the assistant trajectory created together with
    each synthetic user turn.  New fixed-trajectory protocols therefore retain
    that trajectory, but only as the prefix on which source-conditioned Teacher
    soft targets are recomputed.  It is never converted to an answer NLL label.
    Older frozen protocols remain readable without this extra field.
    """
    frozen = Path(config.get('paired_input_dir', config['output']+'/paired_inputs'))/(item['key']+'.json')
    if frozen.is_file():
        value = read(frozen)
        if value['case'] != item['key']: raise ValueError('Paired input belongs to another case')
        if 'dataset_arms' in config and set(value['datasets']) != set(config['dataset_arms']):
            raise ValueError('Frozen data includes a dataset excluded from this experiment')
        result = value['datasets']
        if config.get('fixed_synthetic_answer_trajectories'):
            for splits in result.values():
                for rows in splits.values():
                    if any(not row.get('answer', '').strip() for row in rows):
                        raise ValueError('Frozen fixed-trajectory input lacks an assistant answer')
        apply_specific_evidence_policy(item, result, config)
        return result, value['vanilla_sampled_indices']
    specific = read(item['specific_marker'])
    datasets = config.get('dataset_arms', ('vanilla_syns', 'specific_syns'))
    contexts = {}
    if 'vanilla_syns' in datasets:
        catalog = read(item['original']['source'])
        if isinstance(catalog.get('content'), dict): catalog = catalog['content']
        contexts = {row['context_id']: row['text']
                    for row in catalog.get('original_contexts', catalog.get('contexts', []))}
    result, sampling = {}, {}
    for dataset in datasets:
        result[dataset] = {}
        for n, split in enumerate(('train', 'validation')):
            target = read(specific['paths'][split])['examples']
            paired_counts = item.get('paired_split_counts')
            if paired_counts is not None:
                if (set(paired_counts) != {'train', 'validation'}
                        or any(type(value) is not int or value < 1
                               for value in paired_counts.values())):
                    raise ValueError('Paired split counts must be positive train/validation integers')
                requested = paired_counts[split]
                if len(target) < requested:
                    raise ValueError('Specific dataset is shorter than its frozen paired budget')
                target = target[:requested]
            if dataset == 'specific_syns':
                rows = []
                for index, row in enumerate(target):
                    value = {'question': row['question'],
                        'evidence': '\n\n'.join(chunk['text'] for chunk in
                            sorted(row['evidence']['source_chunks'], key=lambda c: c['span'][0])),
                        'source_row_index': index}
                    if config.get('fixed_synthetic_answer_trajectories'):
                        answer = row.get('answer')
                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError('Answer-guided conversation lacks its answer trajectory')
                        value.update(answer=answer, answer_finished=True,
                            answer_source_quotes=row.get('answer_source_quotes', []),
                            quote_provenance_complete=bool(row.get('quote_provenance_complete')),
                            answer_trajectory_source='answer_guided_answer_first_generation')
                    rows.append(value)
            else:
                original = read(item['original']['files'][n])['conversations']
                original_conversations = None
                if config.get('fixed_synthetic_answer_trajectories'):
                    from cartridges.structs import read_conversations
                    pkl = Path(item['original']['files'][n]).with_suffix('.pkl')
                    if not pkl.is_file():
                        raise FileNotFoundError(f'Missing released conversation tokens: {pkl}')
                    original_conversations = read_conversations(str(pkl))
                    if len(original_conversations) != len(original):
                        raise ValueError('JSON and PKL vanilla conversation counts differ')
                indices = list(range(len(original)))
                random.Random(seed+100+n).shuffle(indices)
                indices = sorted(indices[:len(target)])
                if len(indices) != len(target):
                    raise ValueError('Not enough original questions for a matched comparison')
                sampling[split] = indices
                rows = []
                for index in indices:
                    row = original[index]
                    question = row.get('question') or next(message['content']
                        for message in row['messages'] if message['role'] == 'user')
                    context_id = row.get('source_context_id') or row.get('metadata', {}).get('source_context_id')
                    evidence = contexts.get(context_id)
                    if not isinstance(evidence, str) or not evidence:
                        raise ValueError('Original question lacks its original generation context')
                    value = {'question': question, 'evidence': evidence, 'source_row_index': index}
                    if original_conversations is not None:
                        conversation = original_conversations[index]
                        if (conversation.messages[0].content != question
                                or conversation.messages[-1].role != 'assistant'
                                or conversation.messages[-1].content != row['messages'][-1]['content']):
                            raise ValueError('Released vanilla JSON and token trajectory differ')
                        token_ids = list(map(int, conversation.messages[-1].token_ids))
                        if not token_ids:
                            raise ValueError('Released vanilla answer trajectory is empty')
                        value.update(answer=conversation.messages[-1].content,
                            answer_token_ids=token_ids,
                            answer_trajectory_source='released_Cartridges_assistant_turn')
                    rows.append(value)
            if any(not row['question'].strip() or not row['evidence'].strip() for row in rows):
                raise ValueError('Incomplete source-bound synthetic question')
            if (config.get('fixed_synthetic_answer_trajectories')
                    and any(not row.get('answer', '').strip() for row in rows)):
                raise ValueError('Fixed-trajectory protocol lost a synthetic assistant turn')
            result[dataset][split] = rows
    apply_specific_evidence_policy(item, result, config)
    return result, sampling


def query_relevant_evidence_union(item):
    """Merge already-retrieved guided chunks in original document order.

    This performs no new retrieval and reads no synthetic or benchmark answer.
    It prevents different exercises for one real query from seeing mutually
    incomplete temporal/relational slices of the same evidence chain.
    """
    marker = read(item['specific_marker'])
    chunks = {}
    for split in ('train', 'validation'):
        examples = read(marker['paths'][split])['examples']
        for example in examples:
            for chunk in example['evidence']['source_chunks']:
                index = int(chunk['source_index'])
                span = tuple(map(int, chunk['span']))
                text = chunk['text']
                if len(span) != 2 or not 0 <= span[0] < span[1] or not text.strip():
                    raise ValueError('Invalid guided source chunk')
                value = (span, text)
                if index in chunks and chunks[index] != value:
                    raise ValueError('One source index has inconsistent guided evidence')
                chunks[index] = value
    if not chunks:
        raise ValueError('Query-guided data has no source evidence')
    ordered = sorted(((index, *value) for index, value in chunks.items()),
                     key=lambda value: (value[1][0], value[0]))
    if any(left[1][0] >= right[1][0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Query-guided source spans are not uniquely ordered')
    return '\n\n'.join(value[2] for value in ordered), [value[0] for value in ordered], [value[1] for value in ordered]


def apply_specific_evidence_policy(item, datasets, config):
    policy = config.get('specific_teacher_evidence_policy', 'per_exercise_bound_evidence')
    if policy not in {'per_exercise_bound_evidence', 'query_relevant_union'}:
        raise ValueError(f'Unknown specific evidence policy: {policy}')
    if 'specific_syns' not in datasets or policy == 'per_exercise_bound_evidence':
        return datasets
    evidence, indices, spans = query_relevant_evidence_union(item)
    for rows in datasets['specific_syns'].values():
        for row in rows:
            row['local_evidence'] = row['evidence']
            row['evidence'] = evidence
            row['evidence_policy'] = policy
            row['query_evidence_source_indices'] = indices
            row['query_evidence_source_spans'] = spans
    return datasets


def materialize_training_prompts(case, inputs, config):
    """Attach the exact Student prompt without changing either synthetic set.

    The query-specific questions are part of the current BABILong task, so they
    must retain that task's instructions, demonstrations, and answer format.
    Vanilla Cartridges remains the released raw-conversation control.
    """
    from contextualized.data import task_without_document
    policy = config.get('specific_prompt_policy', 'raw_synthetic_question')
    if policy not in {'raw_synthetic_question',
                      'original_task_wrapper_with_synthetic_question'}:
        raise ValueError(f'Unknown query-specific prompt policy: {policy}')
    for dataset, splits in inputs.items():
        for rows in splits.values():
            for row in rows:
                prompt = row['question']
                if (dataset == 'specific_syns'
                        and policy == 'original_task_wrapper_with_synthetic_question'):
                    prompt = task_without_document(case, row['question'])
                    if (case['document'] in prompt
                            or prompt.count('[The context is provided above.]') != 1
                            or row['question'] not in prompt):
                        raise ValueError('Task-aligned synthetic prompt leaked or lost source data')
                row['training_prompt'] = prompt
                row['training_prompt_sha256'] = sha256(prompt.encode()).hexdigest()
                row['training_prompt_policy'] = (
                    policy if dataset == 'specific_syns' else 'official_raw_conversation')
    return inputs


def critic_task_prompt(case, row, dataset, config, document):
    """Build the source-visible critic task corresponding to the Student prompt."""
    from contextualized.data import task_without_document
    if (dataset == 'specific_syns'
            and config.get('specific_prompt_policy') ==
            'original_task_wrapper_with_synthetic_question'):
        prompt = task_without_document(case, row['question'])
        marker = '[The context is provided above.]'
        if prompt.count(marker) != 1:
            raise ValueError('Expected one source placeholder in BABILong task prompt')
        prompt = prompt.replace(marker, document, 1)
    else:
        from experiments.selective_learning_runtime import SYSTEM_TEMPLATE
        prompt = SYSTEM_TEMPLATE.format(subcorpus=document) + '\n\n' + row['question']
    if prompt.count(document) != 1 or row['question'] not in prompt:
        raise ValueError('Critic task must contain one complete source and its question')
    return prompt


def save_conversation(conversation, path):
    from cartridges.structs import write_conversations
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+f'.{os.getpid()}.tmp.pkl')
    write_conversations([conversation], str(temp)); temp.replace(path)


def load_conversation(path, row, dataset, split):
    from cartridges.structs import read_conversations
    values = read_conversations(str(path))
    if len(values) != 1:
        raise ValueError('Teacher checkpoint must hold one conversation')
    value = values[0]
    if (value.metadata.get('original_question') != row['question']
            or value.metadata.get('dataset_arm') != dataset
            or value.metadata.get('split') != split
            or value.messages[0].content != row.get('training_prompt', row['question'])):
        raise ValueError('Teacher checkpoint differs from the saved synthetic question')
    return value


def prepare_targets(engine, case, inputs, output, config, seed, deadline):
    import torch
    from experiments.syn_sd_factorial_runtime import FullDocumentTeacher
    from experiments.selective_learning_runtime import _conversation
    source_conditioned = config.get('teacher_context_policy') == 'question_source_evidence'
    fixed_trajectories = bool(config.get('fixed_synthetic_answer_trajectories'))
    teacher = None if source_conditioned else FullDocumentTeacher(
        engine, case['document'], config, deadline)
    targets, counts = {}, {'new': 0, 'reused': 0}
    try:
        for dataset, splits in inputs.items():
            targets[dataset] = {}
            for split, rows in splits.items():
                targets[dataset][split] = []
                for index, row in enumerate(rows):
                    if time.time() >= deadline: raise TimeoutError('Teacher target deadline')
                    path = output/'targets'/dataset/split/f'{index:03d}.pkl'
                    reused = path.is_file()
                    if reused:
                        conversation = load_conversation(path, row, dataset, split)
                        expected_context = row['evidence'] if source_conditioned else case['document']
                        from experiments.selective_learning_runtime import SYSTEM_TEMPLATE
                        if conversation.system_prompt != SYSTEM_TEMPLATE.format(
                                subcorpus=expected_context):
                            raise ValueError('Cached Teacher target used a different source context')
                        if (fixed_trajectories
                                and conversation.metadata.get(
                                    'synthetic_answer_trajectory_used') is not True):
                            raise ValueError('Cached Teacher target lost its fixed assistant trajectory')
                        counts['reused'] += 1
                    else:
                        started = time.monotonic()
                        torch.manual_seed(seed+1000*(dataset == 'specific_syns')+100*(split == 'validation')+index)
                        prompt = row.get('training_prompt', row['question'])
                        context = row['evidence'] if source_conditioned else case['document']
                        if fixed_trajectories:
                            from types import SimpleNamespace
                            answer_ids = row.get('answer_token_ids')
                            if answer_ids is None:
                                answer_ids = engine.tokenizer.encode(
                                    row['answer'], add_special_tokens=False)
                                if (row.get('answer_finished', True) and answer_ids
                                        and answer_ids[-1] not in engine.eos_ids):
                                    eos = engine.tokenizer.eos_token_id
                                    if eos is None or eos not in engine.eos_ids:
                                        raise ValueError('Cannot terminate fixed assistant trajectory')
                                    answer_ids.append(int(eos))
                            answer_ids = list(map(int, answer_ids or ()))
                            if not answer_ids:
                                raise ValueError('Fixed assistant trajectory is empty')
                            if len(answer_ids) > config['learning_answer_tokens']:
                                raise ValueError('Fixed assistant trajectory exceeds answer budget')
                            example = engine.teacher_force(SimpleNamespace(
                                question=prompt, answer_ids=answer_ids,
                                finished=answer_ids[-1] in engine.eos_ids), context,
                                prefix=None, topk=config['teacher_topk'])
                        elif source_conditioned:
                            example = engine.generate_teacher(prompt, context,
                                max_new_tokens=config['learning_answer_tokens'],
                                sampling_temperature=config['teacher_sampling_temperature'],
                                topk=config['teacher_topk'])
                        else:
                            example = teacher.generate(prompt,
                                max_new_tokens=config['learning_answer_tokens'],
                                temperature=config['teacher_sampling_temperature'],
                                topk=config['teacher_topk'])
                        conversation = _conversation(engine, prompt, context,
                                                     example, index, row['question'])
                        conversation.metadata.update(dataset_arm=dataset, split=split,
                            teacher_context=('question_source_evidence' if source_conditioned
                                             else 'full_document'),
                            supervision=('unchanged_source_Teacher_sparse_SD' if source_conditioned
                                         else 'same_full_document_teacher_sparse_SD'),
                            teacher_finished=example.finished, source_row_index=row['source_row_index'],
                            synthetic_answer_trajectory_used=fixed_trajectories,
                            answer_trajectory_source=row.get('answer_trajectory_source'),
                            synthetic_answer_labels_used=False, generation_seconds=time.monotonic()-started,
                            training_prompt_policy=row.get('training_prompt_policy'),
                            training_prompt_sha256=row.get('training_prompt_sha256'))
                        save_conversation(conversation, path)
                        counts['new'] += 1
                        del example
                    targets[dataset][split].append(conversation)
                    print(json.dumps({'event': 'factorial_teacher_target', 'case': case['key'],
                        'dataset': dataset, 'split': split, 'index': index, 'reused': reused,
                        'answer_tokens': len(conversation.messages[-1].token_ids)}), flush=True)
    finally:
        if teacher is not None:
            teacher.clear()
        del teacher
        gc.collect(); torch.cuda.empty_cache()
    return targets, counts


def select_targets(engine, case, inputs, targets, output, config, deadline):
    import torch
    from experiments.syn_sd_factorial_runtime import SourceAssessmentGenerator
    from experiments.selective_learning_runtime import assess_teacher_signal, _legacy_helpers, SYSTEM_TEMPLATE
    helpers = _legacy_helpers(config['stack_ttt_root'])
    assessment_generator = SourceAssessmentGenerator(engine, case['document'], config, deadline)
    selected, audits = {}, {}
    try:
        for dataset, splits in inputs.items():
            selected[dataset], audits[dataset] = {}, {}
            for split in ('train', 'validation'):
                selected[dataset][split], audits[dataset][split] = [], []
                for index, (row, conversation) in enumerate(zip(splits[split], targets[dataset][split])):
                    if time.time() >= deadline: raise TimeoutError('Selective SD preparation deadline')
                    path = output/'assessments'/dataset/split/f'{index:03d}.json'
                    candidate = conversation.messages[-1].content
                    if path.is_file():
                        audit = read(path)
                        if audit['question'] != row['question'] or audit['candidate'] != candidate:
                            raise ValueError('Selective checkpoint differs from unchanged Teacher target')
                        if audit['assessment'].get('decision_source') != config.get(
                                'selective_gate', 'task_conditioned_support_required'):
                            raise ValueError('Cached admission belongs to a different Selective SD gate')
                    else:
                        began = time.monotonic()
                        task = critic_task_prompt(case, row, dataset, config, case['document'])
                        assessment = assess_teacher_signal(document=case['document'], question=row['question'],
                            task_prompt=task, evidence=row['evidence'], candidate=candidate,
                            candidate_complete=conversation.metadata['teacher_finished'],
                            generate_assessment=assessment_generator, legacy_helpers=helpers,
                            admission_policy=config.get('selective_gate','task_conditioned_support_required'))
                        audit = {'question_index': index, 'question': row['question'], 'candidate': candidate,
                            'dataset': dataset, 'split': split, 'assessment': assessment,
                            'teacher_targets_unchanged': True, 'seconds': time.monotonic()-began,
                            'benchmark_answers_used': False, 'synthetic_answer_labels_used': False}
                        write(path, audit)
                    if audit['assessment']['apply_update']:
                        selected[dataset][split].append(conversation)
                    audits[dataset][split].append({'question_index': index,
                        'apply_update': audit['assessment']['apply_update'],
                        'reason': audit['assessment']['reason'], 'path': str(path)})
                    print(json.dumps({'event': 'factorial_selective_signal', 'case': case['key'],
                        'dataset': dataset, 'split': split, 'index': index,
                        'accepted': audit['assessment']['apply_update']}), flush=True)
        transport = {'source_prefix_reused_calls': assessment_generator.cached_calls,
                     'fresh_short_context_calls': assessment_generator.fresh_calls}
    finally:
        assessment_generator.clear(); del assessment_generator; gc.collect(); torch.cuda.empty_cache()
    return selected, audits, transport


def process_case(engine, case, item, config, rank, deadline):
    import torch
    from contextualized.data import task_without_document
    from contextualized.cache_io import save_cartridge, load_cartridge
    from experiments.run_contextual_learning import train_arm
    from experiments.selective_learning_runtime import FrozenAssessmentGenerator
    from experiments.syn_sd_factorial_runtime import resident
    root = Path(config['output']); output = root/'cases'/case['key']
    seed = (config['seed']+zlib.crc32(case['key'].encode())) % (2**31)
    started = time.monotonic()
    inputs, sampling = input_rows(item, config, seed)
    materialize_training_prompts(case, inputs, config)
    manifest = {'case': case['key'], 'seed': seed, 'dataset_counts': {
        dataset: {split: len(rows) for split, rows in splits.items()} for dataset, splits in inputs.items()},
        'vanilla_sampled_source_indices': sampling, 'source': item,
        'teacher_context': config['teacher_context'], 'synthetic_answer_labels_used': False,
        'synthetic_answer_trajectories_used': bool(
            config.get('fixed_synthetic_answer_trajectories')),
        'synthetic_answer_hard_label_loss': False,
        'specific_prompt_policy': config.get('specific_prompt_policy','raw_synthetic_question'),
        'specific_teacher_evidence_policy':config.get(
            'specific_teacher_evidence_policy','per_exercise_bound_evidence'),
        'benchmark_answers_used': False}
    if (output/'input.json').is_file() and read(output/'input.json') != manifest:
        raise ValueError('Saved paired input manifest differs from this request')
    write(output/'input.json', manifest)
    baseline_path = output/'baselines.json'
    if config.get('teacher_backend') == 'vllm_subprocess':
        import subprocess, sys
        engine.student.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        prepare_path=output/'teacher_prepare.log'
        prepare_log=prepare_path.open('a')
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('prepare_sd_teacher_vllm.py')),
            '--config', config['_config_path'], '--case', case['key']], start_new_session=True,
            stdout=prepare_log,stderr=subprocess.STDOUT)
        try:
            status = child.wait(timeout=max(1, deadline-time.time()-5))
            if status:
                raise TeacherPreparationError(f'Teacher preparation exited with status {status}; log: {prepare_path}')
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 9); child.wait(); raise TimeoutError('Teacher preparation deadline')
        finally:
            try: os.killpg(child.pid, 15)
            except ProcessLookupError: pass
            prepare_log.close()
        baselines = read(baseline_path); targets, selected = {}, {}
        supervision = read(output/'supervision_summary.json')
        for dataset, splits in inputs.items():
            targets[dataset] = {split: [load_conversation(output/'targets'/dataset/split/f'{index:03d}.pkl',
                row, dataset, split) for index, row in enumerate(rows)] for split, rows in splits.items()}
            selected[dataset] = {}
            for split,audit_key in (('train','gate_audits'),
                                    ('validation','validation_gate_audits')):
                admitted = {entry['question_index'] for entry in supervision[audit_key][dataset]
                            if entry['apply_update']}
                selected[dataset][split] = [value for index,value in enumerate(targets[dataset][split])
                                            if index in admitted]
        target_counts = supervision['target_cache']
        gate_audits = {dataset:{'train':supervision['gate_audits'][dataset],
                                'validation':supervision['validation_gate_audits'][dataset]}
                       for dataset in inputs}
        transport = supervision['transport']
    else:
        resident(engine, 'teacher')
        if baseline_path.is_file():
            baselines = read(baseline_path)
        else:
            generate = FrozenAssessmentGenerator(engine, prefill_tokens=config['prefill_tokens'])
            full = generate(case['prompt'], config['eval_tokens'])
            bare = generate(task_without_document(case), config['eval_tokens'])
            baselines = {'full_context_teacher': full['prediction'], 'no_context': bare['prediction'],
                'full_teacher_prompt': 'original_benchmark_task_and_full_document',
                'full_teacher_complete': full['complete'], 'no_context_complete': bare['complete']}
            write(baseline_path, baselines)
        targets, target_counts = prepare_targets(engine, case, inputs, output, config, seed, deadline)
        selected, gate_audits, transport = select_targets(engine, case, inputs, targets, output, config, deadline)
    write(output/'supervision_summary.json', {'target_cache': target_counts,
        'train_counts': {dataset: {'vanilla_sd': len(targets[dataset]['train']),
                                  'selective_sd': len(selected[dataset]['train'])} for dataset in inputs},
        'validation_counts': {dataset: {'vanilla_sd': len(targets[dataset]['validation']),
                                  'selective_sd': len(selected[dataset]['validation'])} for dataset in inputs},
        'gate': config['selective_gate'],
        'gate_audits': {dataset: value['train'] for dataset,value in gate_audits.items()},
        'validation_gate_audits': {dataset: value['validation'] for dataset,value in gate_audits.items()},
        'transport': transport,
        'heldout_policy': config['validation_policy'], 'teacher_targets_unchanged': True})
    resident(engine, 'student')
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    initial_path = output/'initial.safetensors'
    if initial_path.is_file(): initial, _ = load_cartridge(initial_path, device=engine.device)
    else:
        initial = engine.initial_cache(case['document'])
        save_cartridge(initial, initial_path, config['model'])
    query = task_without_document(case)
    if 'initial_kv' not in baselines:
        baselines['initial_kv'] = engine.generate_student(query, initial, max_new_tokens=config['eval_tokens'])[0]
        initial.clear(); write(baseline_path, baselines)
    results = {}
    for dataset in config.get('dataset_arms', ('vanilla_syns', 'specific_syns')):
        for sd in config.get('sd_arms', ('oracle_sd', 'selective_sd')):
            name = dataset+'_'+sd; arm_path = output/'arms'/(name+'.json')
            if arm_path.is_file():
                results[name] = read(arm_path)
                continue
            if time.time() >= deadline: raise TimeoutError('Native KV adaptation deadline')
            arm_selected = selected
            tracin_audit = None
            if sd == 'selective_sd' and config.get('tracin_selective_sd'):
                reference_name = dataset + '_vanilla_sd'
                if reference_name not in results:
                    raise ValueError('Vanilla SD must precede TracIn-Selective SD')
                candidates = selected[dataset]['train']
                tracin_kept, tracin_audit = tracin_selective_view(
                    engine, initial, results[reference_name]['cartridge'],
                    targets[dataset]['train'], candidates, config, seed)
                arm_selected = {
                    dataset: {
                        'train': tracin_kept,
                        # Validation remains the ordinary trainer's existing
                        # checkpoint split; it never enters TracIn scoring.
                        'validation': selected[dataset]['validation'],
                    }
                }
                write(output/'tracin'/f'{dataset}.json', tracin_audit)
            views = training_views(targets, arm_selected, dataset, sd, config)
            values = views['values']
            consolidation = views['consolidation']
            heldout = views['heldout']
            fallback = views['fallback']
            validation_fallback = views['validation_fallback']
            selective_consolidation = views['selective_consolidation']
            began = time.monotonic()
            arm_config = arm_training_config(config, results, name)
            reference = config.get('matched_update_reference_arm')
            if (sd == 'selective_sd'
                    and config.get('selective_match_vanilla_packed_sequences')):
                vanilla_name = dataset+'_vanilla_sd'
                if vanilla_name not in results:
                    raise ValueError('Vanilla SD must precede compute-matched Selective SD')
                arm_config['train_batch_size'] = \
                    results[vanilla_name]['training']['packed_sequences_per_update']
            exact_tracin_fallback = bool(
                sd == 'selective_sd'
                and config.get('tracin_selective_sd')
                and fallback)
            if exact_tracin_fallback:
                vanilla_name = dataset + '_vanilla_sd'
                if vanilla_name not in results:
                    raise ValueError('TracIn fallback requires completed Vanilla SD')
                cache, _ = load_cartridge(
                    results[vanilla_name]['cartridge'], device=engine.device)
                reference_training = results[vanilla_name]['training']
                training = {
                    'objective': 'upstream_sparse_teacher_distillation',
                    'direct_answer_nll': False,
                    'train_conversations': len(values),
                    'heldout_conversations': len(heldout),
                    'updates': reference_training['updates'],
                    'returned_checkpoint_step': reference_training.get(
                        'returned_checkpoint_step', reference_training['updates']),
                    'returned_checkpoint_training_updates': reference_training.get(
                        'returned_checkpoint_training_updates',
                        reference_training['updates']),
                    'final_heldout_kd_loss': reference_training.get(
                        'final_heldout_kd_loss'),
                    'packed_sequences_per_update': reference_training.get(
                        'packed_sequences_per_update'),
                    'stop_reason': 'all_TracIn_scores_nonpositive_exact_Vanilla_fallback',
                    'returned_checkpoint_source': vanilla_name,
                    'exact_trained_vanilla_checkpoint_reused': True,
                    'history': [],
                    'benchmark_answers_used': False,
                }
            else:
                cache, training = train_arm(engine, initial, values, heldout, arm_config, seed,
                    consolidation_conversations=(consolidation
                        if sd == 'selective_sd' and config.get('selective_consolidation_schedule')
                        else None))
            if (reference and reference != name
                    and config.get('matched_update_unit') == 'packed_optimizer_steps'):
                training.update(
                    matched_update_reference_arm=reference,
                    matched_update_unit='packed_optimizer_steps',
                    matched_reference_updates=results[reference]['training']['updates'],
                    one_packed_sequence_per_optimizer_update=True,
                    checkpoint_evaluation_boundaries=[
                        max(1, results[reference]['training']['updates'] // 2),
                        results[reference]['training']['updates'],
                    ],
                )
            training['all_rejected_original_SD_fallback']=fallback
            training['all_rejected_validation_original_SD_fallback']=validation_fallback
            if tracin_audit is not None:
                training['tracin_selective_sd'] = tracin_audit
            if sd == 'selective_sd' and config.get('selective_match_vanilla_packed_sequences'):
                training['compute_matched_to_vanilla_packed_sequences_per_update'] = \
                    results[dataset+'_vanilla_sd']['training']['packed_sequences_per_update']
                training['retained_sequences_repeated_when_needed'] = True
            if sd == 'selective_sd' and config.get('selective_checkpoint_envelope'):
                reference_name = dataset+'_vanilla_sd'
                if reference_name not in results:
                    raise ValueError('Vanilla SD must finish before its Selective checkpoint envelope')
                candidate_loss = training.get('final_heldout_kd_loss')
                if config.get('selective_checkpoint_reference_scope') == \
                        'same_selected_heldout':
                    reference_loss = saved_cartridge_heldout_loss(
                        engine, results[reference_name]['cartridge'], heldout,
                        arm_config, seed)
                    envelope_metric = 'same_selected_heldout_sparse_Teacher_KD'
                else:
                    reference_loss = results[reference_name]['training'].get(
                        'final_heldout_kd_loss')
                    envelope_metric = 'same_unfiltered_heldout_sparse_Teacher_KD'
                if candidate_loss is None or reference_loss is None:
                    raise ValueError('Checkpoint envelope requires the same nonempty heldout targets')
                tolerance = float(config.get('selective_checkpoint_tolerance', 0.0))
                relative_tolerance = float(config.get(
                    'selective_checkpoint_relative_tolerance', 0.0))
                consolidation_start = arm_config.get(
                    'selective_consolidation_start_step')
                candidate_contains_selective_update = bool(
                    not fallback
                    and (consolidation_start is None
                         or int(training.get('returned_checkpoint_step', 0)) >=
                            int(consolidation_start)))
                use_candidate, required_delta = use_selective_checkpoint(
                    candidate_loss, reference_loss, tolerance=tolerance,
                    relative_tolerance=relative_tolerance,
                    has_accepted_signal=candidate_contains_selective_update)
                # When the gate admits nothing, "original SD" means the
                # already-trained Vanilla checkpoint, not a second training
                # run over the same data in a different shuffle order.  The
                # latter can create a spurious Selective-SD difference even
                # though selection supplied no signal.
                training.update(selective_candidate_heldout_kd_loss=candidate_loss,
                    vanilla_reference_heldout_kd_loss=reference_loss,
                    checkpoint_envelope_required_absolute_improvement=required_delta,
                    checkpoint_envelope_relative_tolerance=relative_tolerance,
                    checkpoint_envelope_metric=envelope_metric,
                    checkpoint_envelope_selected=('selective_candidate' if use_candidate
                                                  else 'trained_vanilla_SD'),
                    selective_candidate_contains_selective_update=(
                        candidate_contains_selective_update),
                    selective_consolidation_start_step=consolidation_start,
                    no_accepted_signal_forces_trained_vanilla_checkpoint=bool(fallback),
                    checkpoint_envelope_benchmark_answers_used=False,
                    checkpoint_envelope_hard_answer_NLL_used=False)
                if not use_candidate:
                    candidate_updates = training['updates']
                    cache.clear(); del cache
                    cache, _ = load_cartridge(results[reference_name]['cartridge'],
                                              device=engine.device)
                    training['returned_checkpoint_training_updates'] = \
                        results[reference_name]['training']['updates']
                    training['selective_candidate_updates'] = candidate_updates
                    training['returned_checkpoint_source'] = reference_name
                else:
                    training['returned_checkpoint_training_updates'] = training['updates']
                    training['selective_candidate_updates'] = training['updates']
                    training['returned_checkpoint_source'] = name
            prediction, tokens = engine.generate_student(query, cache, max_new_tokens=config['eval_tokens'])
            cache.clear()
            delta_sq = 0.
            for trained, base in zip(cache.parameters(), initial.parameters()):
                if trained.requires_grad:
                    delta_sq += float((trained.detach().float()-base.detach().float()).square().sum())
            cartridge = output/'cartridges'/(name+'.safetensors')
            save_cartridge(cache, cartridge, config['model'])
            result = {'case': case['key'], 'dataset': dataset, 'sd': sd, 'status': 'completed',
                'prediction': prediction, 'generated_token_ids': list(map(int, tokens)),
                'training': training, 'native_kv_parameter_change_l2': delta_sq**.5,
                'native_kv_only': True, 'backbone_frozen': True,
                'inference_reads_original_document': False, 'cartridge': str(cartridge),
                'seconds': time.monotonic()-began, 'benchmark_answers_used': False,
                'synthetic_answer_labels_used': False,
                'synthetic_answer_trajectories_used': bool(
                    config.get('fixed_synthetic_answer_trajectories')),
                'synthetic_answer_hard_label_loss': False,
                'no_accepted_signal': sd=='selective_sd' and not bool(
                    arm_selected[dataset]['train']),
                'answer_always_generated': True}
            write(arm_path, result); results[name] = result
            print(json.dumps({'event': 'factorial_arm_complete', 'case': case['key'], 'arm': name,
                'updates': training['updates'], 'seconds': result['seconds']}), flush=True)
            del cache; gc.collect(); torch.cuda.empty_cache()
    initial.clear(); del initial
    record = {'case': case['key'], 'query': case['query'], 'task': item['task'], 'length': item['length'],
        'status': 'completed', 'protocol': config['protocol'], 'rank': rank,
        'predictions': {name: arm['prediction'] for name, arm in results.items()},
        'baselines': baselines, 'arms': results, 'inputs': manifest,
        'supervision_summary': str(output/'supervision_summary.json'),
        'seconds': time.monotonic()-started, 'completed_epoch': time.time(),
        'benchmark_answers_used_for_training': False, 'native_kv_only': True,
        'analysis_split':item.get('analysis_split','unspecified'),
        'gpu': torch.cuda.get_device_name()}
    write(root/'results/cases'/(case['key']+'.json'), record)
    return record


def run(config, rank):
    from contextualized.data import read_indexed_case
    from experiments.syn_sd_factorial_runtime import load_engine
    root = Path(config['output']); source = Path(config['dataset_root'])
    readiness = read(source/'synthetic_dataset_export_progress.json')
    if not readiness.get('complete') or readiness['counts']['completed_data_cases'] != 1000:
        raise ValueError('The 1000-case synthetic dataset must finish before SD starts')
    if not 0 <= rank < config['workers'] <= 12: raise ValueError('Invalid GPU worker rank')
    allocation_deadline = float(os.environ.get(
        'PAIR_DEADLINE_EPOCH', config['deadline_epoch']))
    deadline = (allocation_deadline if config.get('deadline_from_slurm_allocation')
                else min(config['deadline_epoch'], allocation_deadline))
    plan = read(config['source_plan']); index = {row['key']: row for row in read(plan['source_index'])}
    engine = None; done, failed = [], []
    for item in config['cases']:
        key = item['key']; result = root/'results/cases'/(key+'.json')
        if result.is_file(): continue
        if time.time() >= deadline-60: break
        if item.get('analysis_split')=='holdout' and config.get('phase_control_path'):
            while read(config['phase_control_path'])['phase']=='development' and time.time()<deadline-60:
                time.sleep(5)
            if time.time()>=deadline-60 or read(config['phase_control_path'])['phase']=='stop':break
        lock_path = root/'claims'/(key+'.lock'); lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open('a+')
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        try:
            if result.is_file() or (root/'results/failures'/(key+'.json')).is_file(): continue
            if engine is None: engine = load_engine(config)
            case = {**read_indexed_case(plan, index, key), 'task': item['task'], 'length': item['length']}
            print(json.dumps({'event': 'factorial_case_started', 'case': key, 'rank': rank}), flush=True)
            process_case(engine, case, item, config, rank, deadline); done.append(key)
        except TimeoutError as error:
            write(root/'results/partial'/(key+'.json'), {'case': key, 'error': str(error),
                'status': 'deadline_checkpoint', 'rank': rank, 'epoch': time.time()})
            break
        except TeacherPreparationError as error:
            failed.append(key)
            write(root/'results/partial'/(key+'.json'),{'case':key,'error':str(error),
                'status':'retryable_teacher_process_failure','rank':rank,'epoch':time.time(),
                'case_not_excluded_from_other_workers':True})
            print(traceback.format_exc(),flush=True)
            break
        except Exception as error:
            failed.append(key)
            write(root/'results/failures'/(key+'.json'), {'case': key, 'rank': rank,
                'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'epoch': time.time()})
            print(traceback.format_exc(), flush=True)
            import torch
            if isinstance(error, torch.OutOfMemoryError) or 'CUDA' in str(error): break
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()
        write(root/'results/workers'/f'{rank:02d}.json', {'rank': rank, 'completed': done,
            'failed': failed, 'epoch': time.time(), 'terminal': False})
    write(root/'results/workers'/f'{rank:02d}.json', {'rank': rank, 'completed': done,
        'failed': failed, 'epoch': time.time(), 'terminal': True})
    return 1 if failed else 75 if time.time() >= deadline else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--rank', required=True, type=int)
    a = p.parse_args(); config = read(a.config); config['_config_path'] = a.config
    return run(config, a.rank)


if __name__ == '__main__': raise SystemExit(main())
