"""Optional same-question formatting for the main on-policy SD path."""

import torch


def main_student_suffix(original, teacher, *, align, document_length,
                        max_context, rollout_tokens):
    """Select a question-only suffix; never append the teacher's answer.

    The focused teacher stores independently prefetched selected-document KV
    and a question-only suffix. The full Student can reuse the latter without
    sharing either cache or answer tokens. Practice questions already construct
    their own matched suffixes and are not changed here.
    """
    if type(align) is not bool:
        raise ValueError("Prompt alignment must be boolean")
    suffix = teacher.question_tokens if align else original
    if not isinstance(suffix, torch.Tensor) or suffix.ndim != 1 or not len(suffix):
        raise ValueError("The main Student question suffix must be a nonempty vector")
    if document_length + len(suffix) + rollout_tokens > max_context:
        raise ValueError("The Student question and rollout exceed the model context")
    return suffix
