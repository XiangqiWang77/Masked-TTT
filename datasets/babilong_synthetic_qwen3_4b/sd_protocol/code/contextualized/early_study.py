"""Official sparse Teacher distillation with heldout early stopping; no label NLL."""
from dataclasses import dataclass
import math
import random

import torch
import torch.nn.functional as F

from cartridges.datasets import TrainDataset, qwen_messages_to_element


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
    dataset.config = TrainDataset.Config(data_sources=[], targets='logits', top_k_logits=20,
                                         packed_seq_length=packed_tokens, packing_mode='truncate')
    dataset.tokenizer = engine.tokenizer
    converter = getattr(engine, 'message_converter', qwen_messages_to_element)
    dataset.elements = [converter(c.messages, tokenizer=engine.tokenizer) for c in conversations]
    if not dataset.elements:
        raise ValueError('Cannot train a Cartridge without a Teacher conversation')
    dataset.batches = dataset._prepare_batches(seed)
    return dataset


def distillation_terms(engine, cache, batch):
    # Upstream train.py autocasts both the forward pass and sparse loss. In
    # particular, CUDA autocast computes log_softmax in FP32 for BF16 logits.
    # Disable this CUDA context for CPU-only diagnostics; backward stays outside.
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=torch.device(engine.device).type == 'cuda'):
        output = engine.student(input_ids=batch.input_ids.to(engine.device),
                                seq_ids=batch.element_ids.to(engine.device),
                                position_ids=batch.position_ids.to(engine.device),
                                past_key_values=cache, use_cache=True, mode='train')
        # This is the exact sparse probability-weighted loss used in upstream train.py.
        selected = F.log_softmax(output.logits, dim=-1)[0, batch.topk_token_idxs.to(engine.device)-1,
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


def fit_early(engine, cache, conversations, validation, *, lr=.02, batch_size=4,
              packed_tokens=2048, seed=42, rule=None, log=None):
    rule = rule or StopRule()
    train = make_dataset(engine, conversations, packed_tokens, seed)
    heldout = make_dataset(engine, validation, packed_tokens, seed+1) if validation else None
    parameters = [p for p in cache.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    rng = random.Random(seed)
    order, cursor = [], 0
    history, best_parameters = [], None
    initial = evaluate_distillation(engine, cache, heldout) if heldout else None
    reason = 'maximum_steps' if heldout else 'maximum_steps_no_distinct_heldout'
    for step in range(1, rule.max_steps+1):
        optimizer.zero_grad(set_to_none=True)
        running = 0.
        for _ in range(batch_size):
            if cursor >= len(order):
                order = list(range(len(train)))
                rng.shuffle(order)
                cursor = 0
            batch = train[order[cursor]]
            cursor += 1
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
        row = {'step': step, 'train_kd_loss': running}
        if heldout and (step % rule.eval_every == 0 or step == rule.max_steps):
            metric = evaluate_distillation(engine, cache, heldout)
            row['heldout_kd_loss'] = metric
            improved, stop = rule.observe(step, metric)
            if improved:
                best_parameters = [p.detach().cpu().clone() for p in parameters]
            row.update(best_step=rule.best_step, stale_checks=rule.stale)
            if log:
                log(row)
            history.append(row)
            if stop:
                reason = 'heldout_distillation_plateau'
                break
        else:
            history.append(row)
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
            'updates': history[-1]['step'], 'returned_checkpoint_step': rule.best_step or history[-1]['step'],
            'checkpoint_policy': {'track_best_before_min': rule.track_best_before_min,
                'eligible_checkpoints': 'every_heldout_evaluation' if rule.track_best_before_min else 'evaluations_at_or_after_min_steps',
                'minimum_executed_updates': rule.min_steps,
                'patience_counts_before_min_steps': False},
            'final_heldout_kd_loss': evaluate_distillation(engine, cache, heldout) if heldout else None,
            'stop_reason': reason, 'history': history, 'benchmark_answers_used': False}
