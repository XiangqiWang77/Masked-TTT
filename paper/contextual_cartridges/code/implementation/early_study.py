"""Official sparse Teacher distillation with heldout early stopping; no label NLL."""
from dataclasses import dataclass
from contextlib import nullcontext
import math
import random

import torch
import torch.nn.functional as F

from cartridges.datasets import TrainDataset, qwen_messages_to_element
from contextualized.activation_offload import save_on_cpu_preserve_strides


@dataclass
class StopRule:
    min_steps: int = 32
    max_steps: int = 256
    eval_every: int = 8
    patience: int = 4
    relative_min_delta: float = .005
    absolute_min_delta: float = 1e-5
    best_loss: float = math.inf
    best_step: int = 0
    stale: int = 0
    track_best_before_min: bool = False

    def __post_init__(self):
        if not (1 <= self.min_steps <= self.max_steps) or self.eval_every < 1 or self.patience < 1:
            raise ValueError('Early stopping requires a positive minimum update budget')

    def observe(self, step, loss):
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite heldout distillation loss')
        if step < self.min_steps and not self.track_best_before_min:
            return False, False
        delta = max(self.absolute_min_delta, abs(self.best_loss)*self.relative_min_delta)
        improved = self.best_step == 0 or loss < self.best_loss-delta
        if improved:
            self.best_loss, self.best_step, self.stale = loss, step, 0
        else:
            self.stale += 1
        if step < self.min_steps:
            # Save useful early checkpoints without shortening the minimum
            # update budget or spending patience during that protected period.
            self.stale = 0
            return improved, False
        return improved, self.stale >= self.patience


def fixed_update_rule(updates, *, min_steps, eval_every, track_best_before_min=False):
    """Match executed updates, retaining the same heldout checkpoint selection.

    Patience exceeds the whole run, so the control executes the requested
    budget; eligible checkpoints still start at the query arm's minimum step.
    """
    return StopRule(min_steps=min_steps, max_steps=updates,
                    eval_every=eval_every, patience=updates + 1,
                    track_best_before_min=track_best_before_min)


def make_dataset(engine, conversations, packed_tokens, seed):
    """Use original token framing, sequence packing, and target indexing."""
    dataset = object.__new__(TrainDataset)
    packing_mode = getattr(engine, 'dataset_packing_mode', 'truncate')
    dataset.config = TrainDataset.Config(data_sources=[], targets='logits', top_k_logits=20,
                                         packed_seq_length=packed_tokens, packing_mode=packing_mode)
    dataset.tokenizer = engine.tokenizer
    converter = getattr(engine, 'message_converter', qwen_messages_to_element)
    dataset.elements = [converter(c.messages, tokenizer=engine.tokenizer) for c in conversations]
    if not dataset.elements:
        raise ValueError('Cannot train a Cartridge without a Teacher conversation')
    for index, element in enumerate(dataset.elements):
        if element.topk_token_idxs.numel() == 0:
            raise ValueError(f'Teacher conversation {index} has no soft targets before packing')
        if packing_mode == 'pad' and len(element.input_ids) > packed_tokens:
            raise ValueError(f'Conversation {index} needs {len(element.input_ids)} training '
                             f'tokens, beyond {packed_tokens}; refusing to truncate its targets')
    dataset.batches = dataset._prepare_batches(seed)
    return dataset


def distillation_terms(engine, cache, batch):
    if batch.topk_token_idxs.numel() == 0:
        raise ValueError('A packed conversation has no Teacher targets; check prompt/packing lengths')
    # Upstream train.py autocasts both the forward pass and sparse loss. In
    # particular, CUDA autocast computes log_softmax in FP32 for BF16 logits.
    # Disable this CUDA context for CPU-only diagnostics; backward stays outside.
    offload = (getattr(engine, 'sd_offload_saved_tensors', False)
               and torch.is_grad_enabled() and torch.device(engine.device).type == 'cuda')
    saved_tensors = (save_on_cpu_preserve_strides(pin_memory=True)
                     if offload else nullcontext())
    # Optional CPU activation storage trades PCIe transfers for a smaller
    # backward peak; unpacking also restores the exact strides expected by
    # compiled FlexAttention backward, including transposed and broadcast views.
    with saved_tensors, torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=torch.device(engine.device).type == 'cuda'):
        target_positions = batch.topk_token_idxs.to(engine.device) - 1
        projection_options = {}
        if getattr(engine, 'sd_target_only_logits', False):
            # Use the model's existing output-position selector. Attention and
            # the full vocabulary normalizer at each supervised token are
            # unchanged; only unused prompt/padding LM-head rows are omitted.
            positions, target_positions = torch.unique(target_positions,
                                                        sorted=True, return_inverse=True)
            projection_options['logits_to_keep'] = positions
        output = engine.student(input_ids=batch.input_ids.to(engine.device),
                                seq_ids=batch.element_ids.to(engine.device),
                                position_ids=batch.position_ids.to(engine.device),
                                past_key_values=cache, use_cache=True, mode='train',
                                **projection_options)
        # This is the exact sparse probability-weighted loss used in upstream train.py.
        selected = F.log_softmax(output.logits, dim=-1)[0, target_positions,
                                                        batch.topk_token_ids.to(engine.device)]
        terms = -batch.topk_logprobs.to(engine.device).exp()*selected
    if terms.numel() == 0:
        raise ValueError('A packed conversation has no Teacher targets')
    return terms


@torch.no_grad()
def evaluate_distillation(engine, cache, dataset):
    total, count = 0., 0
    for index in range(len(dataset)):
        terms = distillation_terms(engine, cache, dataset[index])
        total += float(terms.float().sum())
        count += terms.numel()
        cache.clear()
    return total/count


@torch.no_grad()
def training_reference_loss(engine, cache, conversations, *, packed_tokens, seed):
    """Use the same equal-trajectory objective as the TracIn reference."""
    if not conversations:
        raise ValueError('A training reference requires conversations')
    return sum(evaluate_distillation(engine, cache,
        make_dataset(engine, [conversation], packed_tokens, seed + index))
        for index, conversation in enumerate(conversations)) / len(conversations)


def fit_early(engine, cache, conversations, validation, *, lr=.02, batch_size=4,
              packed_tokens=2048, seed=42, rule=None, log=None,
              consolidation_conversations=None, consolidation_start_step=2,
              official_epochs=None, consolidation_start_epoch=2,
              training_monitor_every=0, selective_update_guard=False,
              guard_tolerance=1e-6):
    """Fit native KV with an optional answer-preserving selective second pass.

    When ``consolidation_conversations`` is supplied, step one remains the
    complete original Cartridges SD pass.  Subsequent steps draw only from the
    source-supported subset while keeping the same optimizer, update count,
    and packed-sequence forwards per update.  Cycling the subset concentrates
    the freed budget without ever replacing the first full-data update.
    """
    rule = rule or StopRule()
    if selective_update_guard and (official_epochs is not None or validation):
        raise ValueError('The SD update guard uses fixed steps and no heldout set')
    if selective_update_guard and training_monitor_every < 1:
        raise ValueError('The SD update guard requires a positive monitoring interval')
    if consolidation_start_step < 2:
        raise ValueError('Selective consolidation must follow a positive full-SD warmup')
    if official_epochs is not None:
        if type(official_epochs) is not int or official_epochs < 1:
            raise ValueError('Official epoch training requires a positive epoch count')
        if not 2 <= consolidation_start_epoch <= official_epochs:
            raise ValueError('Selective consolidation must follow a positive full-SD epoch')
    train = make_dataset(engine, conversations, packed_tokens, seed)
    consolidation = (make_dataset(engine, consolidation_conversations,
                                  packed_tokens, seed + 2)
                     if consolidation_conversations else None)
    heldout = make_dataset(engine, validation, packed_tokens, seed+1) if validation else None
    if batch_size == 'full_dataset':
        if official_epochs is None:
            # Legacy complete-pass mode averages a whole packed dataset into
            # one optimizer update.
            batch_size = len(train)
        else:
            # Released-trainer mode steps each packed DatasetBatch separately.
            batch_size = 1
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Training batch size must be positive or full_dataset')
    parameters = [p for p in cache.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    rngs = {
        'full': random.Random(seed),
        'selective': random.Random(seed + 2),
    }
    orders = {'full': [], 'selective': []}
    cursors = {'full': 0, 'selective': 0}
    history, best_parameters = [], None
    initial = evaluate_distillation(engine, cache, heldout) if heldout else None
    reason = 'maximum_steps' if heldout else 'maximum_steps_no_distinct_heldout'

    # Match the released trainer's loop nesting when requested: every packed
    # DatasetBatch performs one optimizer update and an epoch visits every
    # packed batch once.  A selective epoch cycles its admitted subset to the
    # same number of packed-batch updates as the corresponding full epoch, so
    # selection changes supervision rather than compute within a dataset arm.
    if official_epochs is not None:
        epoch_rule = StopRule(
            min_steps=1,
            max_steps=official_epochs,
            eval_every=1,
            patience=official_epochs + 1,
            relative_min_delta=rule.relative_min_delta,
            absolute_min_delta=rule.absolute_min_delta,
            track_best_before_min=True,
        )
        update = 0
        best_update = 0
        for epoch in range(1, official_epochs + 1):
            phase = ('selective'
                     if consolidation is not None and epoch >= consolidation_start_epoch
                     else 'full')
            active = consolidation if phase == 'selective' else train
            # The ordinary arm visits each packed full-data batch once.  The
            # selective arm repeats admitted batches only to match that count.
            updates_this_epoch = len(train)
            for batch_in_epoch in range(1, updates_this_epoch + 1):
                if cursors[phase] >= len(orders[phase]):
                    orders[phase] = list(range(len(active)))
                    rngs[phase].shuffle(orders[phase])
                    cursors[phase] = 0
                batch = active[orders[phase][cursors[phase]]]
                cursors[phase] += 1
                optimizer.zero_grad(set_to_none=True)
                terms = distillation_terms(engine, cache, batch)
                loss = terms.mean()
                if not bool(loss.isfinite()):
                    raise FloatingPointError('Nonfinite upstream distillation loss')
                loss.backward()
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in parameters):
                    raise RuntimeError('Native KV received no distillation gradient')
                if any(p.grad is not None and not bool(p.grad.isfinite().all())
                       for p in parameters):
                    raise FloatingPointError('Nonfinite KV gradient')
                optimizer.step()
                cache.clear()
                update += 1
                row = {
                    'step': update,
                    'epoch': epoch,
                    'batch_in_epoch': batch_in_epoch,
                    'train_kd_loss': float(loss.detach()),
                    'training_phase': phase,
                    'phase_packed_sequences': len(active),
                }
                if batch_in_epoch == updates_this_epoch and heldout:
                    metric = evaluate_distillation(engine, cache, heldout)
                    row['heldout_kd_loss'] = metric
                    improved, _ = epoch_rule.observe(epoch, metric)
                    if improved:
                        best_parameters = [p.detach().cpu().clone() for p in parameters]
                        best_update = update
                    row.update(best_epoch=epoch_rule.best_step,
                               best_step=best_update,
                               stale_checks=epoch_rule.stale)
                if log:
                    log(row)
                history.append(row)
        if best_parameters is not None:
            with torch.no_grad():
                for parameter, saved in zip(parameters, best_parameters):
                    parameter.copy_(saved.to(parameter.device))
        cache.clear()
        if any(p.grad is not None for p in engine.student.parameters()):
            raise RuntimeError('The backbone must remain frozen')
        returned_epoch = epoch_rule.best_step or official_epochs
        returned_update = best_update or update
        return {
            'objective': 'upstream_sparse_teacher_distillation',
            'direct_answer_nll': False,
            'train_conversations': len(conversations),
            'heldout_conversations': len(validation),
            'packed_train_sequences': len(train),
            'initial_heldout_kd_loss': initial,
            'selective_consolidation_conversations': (
                len(consolidation_conversations) if consolidation_conversations else 0),
            'packed_selective_sequences': len(consolidation) if consolidation else 0,
            'training_schedule': (
                'official_packed_batch_epochs_with_source_supported_second_epoch'
                if consolidation is not None else 'official_packed_batch_epochs'),
            'selective_consolidation_start_epoch': (
                consolidation_start_epoch if consolidation is not None else None),
            'packed_sequences_per_update': 1,
            'packed_sequences_per_epoch': len(train),
            'effective_full_dataset_epochs': official_epochs,
            'executed_epochs': official_epochs,
            'updates': update,
            'returned_checkpoint_step': returned_update,
            'returned_checkpoint_epoch': returned_epoch,
            'checkpoint_policy': {
                'track_best_before_min': True,
                'eligible_checkpoints': 'end_of_each_positive_training_epoch',
                'minimum_executed_epochs': 1,
                'zero_update_checkpoint_eligible': False,
            },
            'final_heldout_kd_loss': (
                evaluate_distillation(engine, cache, heldout) if heldout else None),
            'stop_reason': ('official_epoch_budget_with_heldout_checkpoint_selection'
                            if heldout else 'official_epoch_budget'),
            'history': history,
            'benchmark_answers_used': False,
        }

    guard = None
    reference_initial = (training_reference_loss(engine, cache, conversations,
        packed_tokens=packed_tokens, seed=seed) if training_monitor_every else None)
    for step in range(1, rule.max_steps+1):
        phase = ('selective'
                 if consolidation is not None and step >= consolidation_start_step
                 else 'full')
        active = consolidation if phase == 'selective' else train
        if selective_update_guard and phase == 'selective' and guard is None:
            from contextualized.sd_update_guard import SDUpdateGuard
            guard = SDUpdateGuard(parameters, optimizer, step=step-1,
                loss=training_reference_loss(engine, cache, conversations,
                    packed_tokens=packed_tokens, seed=seed), tolerance=guard_tolerance)
        optimizer.zero_grad(set_to_none=True)
        running = 0.
        for _ in range(batch_size):
            if cursors[phase] >= len(orders[phase]):
                orders[phase] = list(range(len(active)))
                rngs[phase].shuffle(orders[phase])
                cursors[phase] = 0
            batch = active[orders[phase][cursors[phase]]]
            cursors[phase] += 1
            terms = distillation_terms(engine, cache, batch)
            loss = terms.mean()/batch_size
            if not bool(loss.isfinite()):
                raise FloatingPointError('Nonfinite upstream distillation loss')
            loss.backward()
            running += float(loss.detach())
            cache.clear()
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters):
            raise RuntimeError('Native KV received no distillation gradient')
        if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in parameters):
            raise FloatingPointError('Nonfinite KV gradient')
        optimizer.step()
        cache.clear()
        row = {'step': step, 'train_kd_loss': running,
               'training_phase': phase,
               'phase_packed_sequences': len(active)}
        if training_monitor_every and (step % training_monitor_every == 0
                                        or step == rule.max_steps):
            metric = training_reference_loss(engine, cache, conversations,
                packed_tokens=packed_tokens, seed=seed)
            row['full_training_reference_loss'] = metric
            if guard is not None:
                row['update_guard'] = guard.observe(step, metric)
                row['full_training_reference_loss'] = guard.loss
                cache.clear()
        if heldout and (step % rule.eval_every == 0 or step == rule.max_steps):
            metric = evaluate_distillation(engine, cache, heldout)
            row['heldout_kd_loss'] = metric
            improved, stop = rule.observe(step, metric)
            if improved:
                best_parameters = [p.detach().cpu().clone() for p in parameters]
            row.update(best_step=rule.best_step, stale_checks=rule.stale)
            history.append(row)
            if log:
                log(row)
            if stop:
                reason = 'heldout_distillation_plateau'
                break
        else:
            history.append(row)
            if log:
                log(row)
    if best_parameters is not None:
        with torch.no_grad():
            for parameter, saved in zip(parameters, best_parameters):
                parameter.copy_(saved.to(parameter.device))
    cache.clear()
    if any(p.grad is not None for p in engine.student.parameters()):
        raise RuntimeError('The backbone must remain frozen')
    return {'objective': 'upstream_sparse_teacher_distillation', 'direct_answer_nll': False,
            'train_conversations': len(conversations), 'heldout_conversations': len(validation),
            'packed_train_sequences': len(train), 'initial_heldout_kd_loss': initial,
            'selective_consolidation_conversations': (
                len(consolidation_conversations) if consolidation_conversations else 0),
            'packed_selective_sequences': len(consolidation) if consolidation else 0,
            'training_schedule': (
                'original_SD_warmup_then_source_supported_consolidation'
                if consolidation is not None else 'ordinary_complete_SD_passes'),
            'selective_consolidation_start_step': (
                consolidation_start_step if consolidation is not None else None),
            'packed_sequences_per_update': batch_size,
            'effective_full_dataset_epochs': history[-1]['step'] if batch_size == len(train) else None,
            'updates': history[-1]['step'],
            'optimizer_update_attempts': history[-1]['step'],
            'effective_updates': guard.effective_updates if guard else history[-1]['step'],
            'update_guard': guard.summary() if guard else None,
            'initial_training_reference_loss': reference_initial,
            'final_training_reference_loss': (training_reference_loss(engine, cache, conversations,
                packed_tokens=packed_tokens, seed=seed) if training_monitor_every else None),
            'returned_checkpoint_step': (guard.last_accepted_step if guard
                else rule.best_step or history[-1]['step']),
            'checkpoint_policy': {'track_best_before_min': rule.track_best_before_min,
                'eligible_checkpoints': 'every_heldout_evaluation' if rule.track_best_before_min else 'evaluations_at_or_after_min_steps',
                'minimum_executed_updates': rule.min_steps,
                'patience_counts_before_min_steps': False},
            'final_heldout_kd_loss': evaluate_distillation(engine, cache, heldout) if heldout else None,
            'stop_reason': reason, 'history': history, 'benchmark_answers_used': False}
