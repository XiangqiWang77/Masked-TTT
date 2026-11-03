"""Distill the current full-context reader into its selected-cache view."""
import torch

from .batched_sd import batched_teacher_logits
from .on_policy import student_rollout
from .optimizer import distribution_kl
from .teacher_targeting import choice_positions


def selected_transfer_loss(engine,full_reader,selected_reader,full_cache,selected_cache,
                           suffix,rollout_limit,*,target_choices=False):
    """Use the selected student's own prefix; stop gradients through the full view."""
    rollout=student_rollout(engine,selected_reader,selected_cache,suffix,rollout_limit)
    positions=choice_positions(engine.tokenizer,rollout.tolist()) if target_choices else []
    indices=(torch.tensor(positions,device=rollout.device,dtype=torch.long)
             if positions else None)
    with torch.no_grad():
        full_logits=batched_teacher_logits(engine,full_reader,full_cache,suffix,rollout,
            checkpoint_layers=False,target_positions=indices)
        target=full_logits.float().softmax(-1).detach()
    selected_logits=batched_teacher_logits(engine,selected_reader,selected_cache,suffix,rollout,
                                            target_positions=indices)
    loss=distribution_kl(selected_logits,target)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite full-to-selected KL")
    return loss,rollout,positions or list(range(len(rollout)))
