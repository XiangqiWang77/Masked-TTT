"""Keep the Teacher's generic instruction while sharing it with the Student."""
from contextlib import contextmanager
from dataclasses import replace

import torch

from masked_ttt import Request


def shared_request(request,instruction):
    """Move an answer-format instruction into both original question suffixes.

    The instruction is the same generic text previously appended by the
    Teacher; it must not contain a Teacher answer or a benchmark reference.
    """
    if not isinstance(instruction,str) or not instruction.strip():
        raise ValueError('A nonempty generic output instruction is required')
    return Request(document=request.document,question=request.question,
        prompt=request.prompt+'\n\nResponse requirements:\n'+instruction,
        answer_prefix=request.answer_prefix,max_new_tokens=request.max_new_tokens)


def shared_config(config):
    """Avoid appending the already-shared instruction a second time."""
    if not config.teacher_response_instruction.strip():
        raise ValueError('The original teacher instruction must be present')
    return replace(config,teacher_response_instruction='')


@contextmanager
def verify_shared_training_suffix():
    """Assert each on-policy teacher score uses the Student's same question."""
    import masked_ttt.pipeline as pipeline
    old_rollout=pipeline.student_rollout;old_score=pipeline.score_student_prefix
    observed=[];pending=None

    def rollout(engine,reader,cache,suffix,limit):
        nonlocal pending
        if pending is not None:raise AssertionError('A student rollout was not scored')
        pending=suffix.detach().clone()
        return old_rollout(engine,reader,cache,suffix,limit)

    def score(engine,reader,teacher,tokens):
        nonlocal pending
        if pending is None or not torch.equal(pending,teacher.question_tokens):
            raise AssertionError('Teacher and Student question suffixes do not match')
        observed.append(dict(question_token_count=len(pending),student_prefix_tokens=len(tokens)))
        pending=None
        return old_score(engine,reader,teacher,tokens)

    pipeline.student_rollout=rollout;pipeline.score_student_prefix=score
    try:yield observed
    finally:
        pipeline.student_rollout=old_rollout;pipeline.score_student_prefix=old_score
