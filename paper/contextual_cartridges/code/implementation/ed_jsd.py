"""Entropy-directed, forward-only selection for native Cartridge SD.

The gate uses the *coarsened* distributions on cached Teacher tokens + rest.
It is not full-vocabulary JSD, a correctness verifier, or a non-regression
guarantee. The training loss remains the released sparse soft-target SD loss.

Trajectories carrying a recorded deterministic Source-Support Gate rejection
(truncated, empty, or refused Teacher answers) are excluded from the admitted
pool before JSD scoring. They still participate in warmup and in the exact
full-data fallback, matching the published two-phase algorithm.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter

import torch
import torch.nn.functional as F


def entropy_bits(probabilities):
    """0 log 0 = 0, including disjoint and deterministic distributions."""
    return -torch.special.xlogy(probabilities, probabilities).sum(-1) / math.log(2.)


def directed_jsd(teacher, student, *, epsilon=1e-12):
    """Normalized distributions -> bounded JSD and relative-entropy score."""
    teacher = teacher.detach().to(device='cpu', dtype=torch.float64)
    student = student.detach().to(device='cpu', dtype=torch.float64)
    if teacher.shape != student.shape or teacher.ndim != 2 or not teacher.numel():
        raise ValueError('Require equally shaped nonempty probability matrices')
    for p in (teacher, student):
        if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
            raise ValueError('Probabilities must be finite and nonnegative')
        if not torch.allclose(p.sum(-1), torch.ones(p.shape[0], dtype=p.dtype),
                              rtol=0, atol=1e-10):
            raise ValueError('JSD requires normalized distributions')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Entropy denominator epsilon must be positive')
    ht, hs = entropy_bits(teacher), entropy_bits(student)
    hm = entropy_bits((teacher + student) * .5)
    # Clip only floating-point cancellation around the mathematical bounds.
    jsd = (hm - (ht + hs) * .5).clamp(0., 1.)
    direction = (hs - ht) / (hs + ht + epsilon)
    return {'jsd_bits': jsd, 'teacher_entropy_bits': ht,
            'student_entropy_bits': hs, 'entropy_direction': direction,
            'directed_score': jsd * direction}


def align_sparse_buckets(token_positions, token_ids, teacher_logp, student_logp):
    """Preserve each retained mass and aggregate, never zero, the omitted tail.

    Inputs are the flattened variable-k targets used by the released trainer.
    Student log probabilities must use its full-vocabulary normalizer.
    """
    positions = torch.as_tensor(token_positions, dtype=torch.long, device='cpu')
    ids = torch.as_tensor(token_ids, dtype=torch.long, device='cpu')
    tp = teacher_logp.detach().to(device='cpu', dtype=torch.float64).exp()
    sp = student_logp.detach().to(device='cpu', dtype=torch.float64).exp()
    if not (positions.ndim == ids.ndim == tp.ndim == sp.ndim == 1
            and len(positions) == len(ids) == len(tp) == len(sp) and len(tp)):
        raise ValueError('Sparse target arrays must have equal positive lengths')
    if not bool(torch.isfinite(tp).all() & torch.isfinite(sp).all()):
        raise ValueError('Nonfinite Teacher/Student probabilities')
    pairs = torch.stack((positions, ids), dim=1)
    if len(torch.unique(pairs, dim=0)) != len(pairs):
        raise ValueError('Duplicate vocabulary entry at a supervised position')
    unique, inverse, counts = torch.unique(positions, sorted=True,
                                          return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    starts = counts.cumsum(0) - counts
    slots = torch.arange(len(order)) - torch.repeat_interleave(starts, counts)
    shape = (len(unique), int(counts.max()) + 1)
    t, s = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
    t[inverse[order], slots] = tp[order]
    s[inverse[order], slots] = sp[order]
    for p in (t, s):
        retained = p.sum(-1)
        if bool((retained > 1. + 1e-5).any()):
            raise ValueError('Sparse cached probability mass exceeds one')
        p[:, -1] = (1. - retained).clamp_min(0.)
        # Only repairs <=1e-5 summation/serialization roundoff; does not
        # renormalize top-k to one or erase its omitted mass.
        p /= p.sum(-1, keepdim=True)
    return unique, t, s


def source_support_admissions(conversations):
    """Read each trajectory's recorded Source-Support Gate decision.

    The gate was computed deterministically when the Teacher trajectory was
    generated. A conversation without a recorded decision is admitted: this
    pre-filter removes only identified degenerate trajectories and never
    invents a judgment that generation did not make.
    """
    admissions = []
    for conversation in conversations:
        metadata = getattr(conversation, 'metadata', None)
        gate = metadata.get('source_gate') if isinstance(metadata, dict) else None
        if isinstance(gate, dict) and 'apply_update' in gate:
            admissions.append({'admitted': bool(gate['apply_update']),
                               'reason': gate.get('reason')})
        else:
            admissions.append({'admitted': True, 'reason': 'no_recorded_source_gate'})
    return admissions


def confidence_decision(teacher_buckets, score, minimum_top1_probability=None):
    """Optional absolute-confidence guard, separate from relative entropy.

    The last column is an aggregate tail, not a candidate word. Exclude it
    when measuring the Teacher's top-token probability. This is a confidence
    heuristic, not an answer-correctness test.
    """
    if teacher_buckets.ndim != 2 or teacher_buckets.shape[1] < 2:
        raise ValueError('Expected retained vocabulary tokens and a rest bucket')
    top1 = teacher_buckets[:, :-1].max(-1).values
    confidence = float(top1.clamp_min(1e-300).log().mean().exp())
    if minimum_top1_probability is not None:
        if not 0 < minimum_top1_probability < 1:
            raise ValueError('Teacher confidence threshold must lie in (0, 1)')
    passed = minimum_top1_probability is None or confidence >= minimum_top1_probability
    return dict(keep=bool(score > 0 and passed),
                teacher_top1_geometric_mean=confidence,
                teacher_confidence_passed=passed)


@torch.no_grad()
def score_trajectories(engine, cache, conversations, *, packed_tokens, seed,
                       deadline=math.inf, minimum_teacher_top1_probability=None):
    """One forward-only sweep at the frozen post-warmup KV state."""
    from contextualized.early_study import make_dataset
    admissions = source_support_admissions(conversations)
    rows = []
    started = time.monotonic()
    for index, conversation in enumerate(conversations):
        if time.time() >= deadline:
            raise TimeoutError('ED-JSD scoring reached the allocation deadline')
        if not admissions[index]['admitted']:
            rows.append({'index': index, 'score': None, 'keep': False, 'scored': False,
                         'source_admitted': False,
                         'source_gate_reason': admissions[index]['reason']})
            continue
        data = make_dataset(engine, [conversation], packed_tokens, seed + index)
        if len(data) != 1:
            raise ValueError('Each gate trajectory must be scored completely once')
        batch = data[0]
        absolute = batch.topk_token_idxs.to(engine.device) - 1
        positions, inverse = torch.unique(absolute, sorted=True, return_inverse=True)
        cache.clear()
        try:
            # mode=generate chooses the bounded no-grad FlexAttention tiles.
            # Input framing, packed sequence masks, positions, and KV match SD.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                   enabled=torch.device(engine.device).type == 'cuda'):
                output = engine.student(input_ids=batch.input_ids.to(engine.device),
                    seq_ids=batch.element_ids.to(engine.device),
                    position_ids=batch.position_ids.to(engine.device),
                    past_key_values=cache, use_cache=True, mode='generate',
                    logits_to_keep=positions)
            # Avoid a full [sequence, vocabulary] FP32 temporary. Only score
            # small rows of the already selected LM-head output at a time.
            gathered = torch.empty_like(batch.topk_logprobs, device='cpu', dtype=torch.float32)
            for begin in range(0, len(positions), 32):
                end = min(begin + 32, len(positions))
                mask = (inverse >= begin) & (inverse < end)
                logp = F.log_softmax(output.logits[0, begin:end].float(), dim=-1)
                values = logp[inverse[mask] - begin, batch.topk_token_ids.to(engine.device)[mask]]
                gathered[mask.cpu()] = values.cpu()
                del logp, values
            _, tp, sp = align_sparse_buckets(batch.topk_token_idxs, batch.topk_token_ids,
                                             batch.topk_logprobs, gathered)
            metrics = directed_jsd(tp, sp)
            score = float(metrics['directed_score'].mean())
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite ED-JSD score')
            decision = confidence_decision(tp, score, minimum_teacher_top1_probability)
            rows.append({'index': index, 'score': score, **decision,
                'scored': True, 'source_admitted': True,
                'tokens': len(tp), 'mean_jsd_bits': float(metrics['jsd_bits'].mean()),
                'mean_teacher_bucket_entropy_bits': float(metrics['teacher_entropy_bits'].mean()),
                'mean_student_bucket_entropy_bits': float(metrics['student_entropy_bits'].mean()),
                'mean_teacher_rest_mass': float(tp[:, -1].mean()),
                'mean_student_rest_mass': float(sp[:, -1].mean()),
                'positive_tokens': int((metrics['directed_score'] > 0).sum()),
                'negative_tokens': int((metrics['directed_score'] < 0).sum())})
            del output, gathered
        finally:
            cache.clear()
    scored = sum(bool(row.get('scored')) for row in rows)
    kept = sum(row['keep'] for row in rows)
    return {'rule': 'mean_base2_coarsened_JSD_times_relative_bucket_entropy_difference_gt_zero',
            'support': 'retained_teacher_tokens_plus_one_rest_bucket',
            'source_support_gate': 'recorded_structural_gate_rejections_excluded_before_scoring',
            'full_vocabulary_JSD': False, 'full_vocabulary_entropy': False,
            'cross_trajectory_comparison': False, 'correctness_guaranteed': False,
            'selection_backward_passes': 0, 'selection_forward_passes': scored,
            'seconds': time.monotonic() - started, 'threshold': 0., 'epsilon': 1e-12,
            'minimum_teacher_top1_probability': minimum_teacher_top1_probability,
            'teacher_confidence_measure': 'geometric_mean_retained_top1_probability_excluding_rest',
            'candidates': scored, 'source_gate_rejected': len(rows) - scored,
            'kept': kept, 'removed': scored - kept, 'trajectories': rows,
            'benchmark_answers_used': False}


def update_budget(dataset, config):
    fixed = config.get('fixed_updates')
    total = int(fixed if fixed is not None else len(dataset) * config['official_epochs'])
    if total < 2:
        raise ValueError('Warmup plus consolidation requires at least two SD updates')
    return total


def fit_ed_jsd(engine, cache, conversations, config, seed, *, selective=True,
               deadline=math.inf, progress=None):
    """Exactly T SD updates, one Adam, one forward-only gate after h updates.

    An empty (or complete) admitted set keeps the *existing* full-data sampler
    and optimizer, giving exactly the ordinary trajectory under deterministic
    evaluation-mode forwards. No heldout data, answer NLL, or refits are used.
    """
    from contextualized.early_study import make_dataset, distillation_terms
    full = make_dataset(engine, conversations, config['packed_tokens'], seed)
    total = update_budget(full, config)
    warmup = int(config.get('ed_jsd_warmup_steps', total // 2))
    if not 1 <= warmup < total:
        raise ValueError('Warmup must leave positive consolidation steps')
    if any(p.requires_grad for p in engine.student.parameters()):
        raise ValueError('Only native KV may be trainable')
    parameters = [p for p in cache.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('Native KV has no trainable parameters')
    accumulation = int(config.get('gradient_accumulation_steps', 1))
    lr_warmup = int(config.get('lr_warmup_steps', 0))
    if accumulation < 1 or not 0 <= lr_warmup <= total:
        raise ValueError('Invalid accumulated batch or learning-rate warmup')
    if config['learning_rate'] <= 0 or not math.isfinite(config['learning_rate']):
        raise ValueError('Learning rate must be finite and positive')
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'],
                                 foreach=config.get('optimizer_foreach', None))
    rngs = {'full': random.Random(seed), 'selected': random.Random(seed + 2)}
    orders, cursors = {'full': [], 'selected': []}, {'full': 0, 'selected': 0}
    empty_policy = config.get('ed_jsd_empty_policy', 'continue_vanilla')
    if empty_policy not in ('continue_vanilla', 'retain_warmup'):
        raise ValueError('Unknown empty JSD consolidation policy')
    selected, audit, history = None, None, []
    started = time.monotonic()
    for step in range(1, total + 1):
        if time.time() >= deadline:
            raise TimeoutError('ED-JSD training reached the allocation deadline')
        if selective and step == warmup + 1:
            # Gradients from the final warmup step must not survive scoring.
            optimizer.zero_grad(set_to_none=True)
            audit = score_trajectories(engine, cache, conversations,
                packed_tokens=config['packed_tokens'], seed=seed, deadline=deadline,
                minimum_teacher_top1_probability=config.get('minimum_teacher_top1_probability'))
            keep = [row['index'] for row in audit['trajectories'] if row['keep']]
            if 0 < len(keep) < len(conversations):
                selected = make_dataset(engine, [conversations[i] for i in keep],
                                        config['packed_tokens'], seed + 2)
            if progress:
                progress({'event': 'ed_jsd_gate', 'step': warmup,
                          'kept': audit['kept'], 'removed': audit['removed'],
                          'source_gate_rejected': audit.get('source_gate_rejected', 0),
                          'selection_seconds': audit['seconds']})
            if not keep and empty_policy == 'retain_warmup':
                # Do not quietly train on trajectories just rejected by the
                # confidence guard. Warmup already made real KV updates.
                break
        phase = 'selected' if selected is not None else 'full'
        active = selected if selected is not None else full
        indices = []
        for _ in range(accumulation):
            if cursors[phase] >= len(orders[phase]):
                orders[phase] = list(range(len(active)))
                rngs[phase].shuffle(orders[phase])
                cursors[phase] = 0
            indices.append(orders[phase][cursors[phase]])
            cursors[phase] += 1
        lr = config['learning_rate'] * (min(1., step / lr_warmup) if lr_warmup else 1.)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        # The model is frozen and in eval mode. Repeated indices within a
        # virtual batch have identical gradients, so weight one forward by
        # its multiplicity rather than waste memory/compute on duplicates.
        for index, multiplicity in Counter(indices).items():
            if time.time() >= deadline:
                raise TimeoutError('ED-JSD accumulated update reached its deadline')
            cache.clear()
            terms = distillation_terms(engine, cache, active[index])
            loss = terms.mean() * (multiplicity / accumulation)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite original SD loss')
            loss.backward()
            loss_value += float(loss.detach())
            cache.clear()
            del terms, loss
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError('Nonfinite native KV gradient')
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters):
            raise RuntimeError('Native KV received no SD gradient')
        gradient_norm = None
        if config.get('max_grad_norm') is not None:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters,
                config['max_grad_norm'], error_if_nonfinite=True, foreach=False))
        optimizer.step()
        cache.clear()
        row = {'step': step, 'train_kd_loss': loss_value,
               'learning_rate': lr, 'effective_packed_batch_size': accumulation,
               'unique_microbatches': len(set(indices)),
               'gradient_norm_before_clip': gradient_norm,
               'training_phase': ('warmup' if selective and step <= warmup else phase),
               'phase_packed_sequences': len(active)}
        history.append(row)
        if progress:
            progress(row)
    optimizer.zero_grad(set_to_none=True)
    if any(p.grad is not None for p in engine.student.parameters()):
        raise RuntimeError('Backbone received a gradient')
    cache.clear()
    actual = len(history)
    stopped_at_warmup = actual < total
    return {'objective': 'upstream_sparse_teacher_distillation', 'direct_answer_nll': False,
        'selection': 'entropy_directed_coarsened_JSD' if selective else 'none',
        'train_conversations': len(conversations), 'heldout_conversations': 0,
        'packed_train_sequences': len(full), 'packed_sequences_per_update': accumulation,
        'equivalent_full_data_passes': actual * accumulation / len(full),
        'learning_rate': config['learning_rate'], 'lr_warmup_steps': lr_warmup,
        'updates': actual, 'planned_updates': total, 'returned_checkpoint_step': actual,
        'warmup_steps': warmup if selective else None,
        'consolidation_steps': actual - warmup if selective else None,
        'training_schedule': 'warmup_then_ED_JSD' if selective else 'ordinary_SD',
        'ed_jsd': audit, 'selection_backward_passes': 0,
        'empty_gate_policy': empty_policy,
        'all_rejected_original_SD_fallback': bool(audit and audit['kept'] == 0 and not stopped_at_warmup),
        'exact_vanilla_continuation': bool(audit and audit['kept'] in (0, len(conversations)) and not stopped_at_warmup),
        'optimizer_reinitialized_after_warmup': False,
        'seconds': time.monotonic() - started, 'history': history,
        'benchmark_answers_used': False,
        'stop_reason': 'no_admissible_consolidation_retain_warmup' if stopped_at_warmup else 'fixed_original_SD_update_budget'}
