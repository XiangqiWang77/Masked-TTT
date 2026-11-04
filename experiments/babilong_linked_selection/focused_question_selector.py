"""Exploratory text-only routing of the existing same-budget selector."""
import re

from experiments.babilong_linked_selection.question_overlap_selector import (
    SIMPLE_WHERE_PROPER, question_overlap)
from masked_ttt.linked_selection import choose


NAME = r"[A-Z][A-Za-z'-]{2,}"
CARRY = re.compile(r"(?i:what\s+is)\s+" + NAME + r"\s+(?i:carrying)\s*\?\s*")
GIVE = (
    re.compile(r"(?i:what\s+did)\s+" + NAME + r"\s+(?i:give)\b[^?]{0,100}\?\s*"),
    re.compile(r"(?i:who\s+did)\s+" + NAME + r"\s+(?i:give)\b[^?]{0,100}\?\s*"),
    re.compile(r"(?i:who\s+gave)\b[^?]{1,100}\?\s*"),
)


def focused_question_overlap(document, question, spans, records, fraction=.1, *, include_give=False):
    """Use query text and cached scores only; never receive a task or answer label."""
    if type(include_give) is not bool:
        raise ValueError("include_give must be boolean")
    text = question.strip()
    eligible = (SIMPLE_WHERE_PROPER.fullmatch(text) is not None or CARRY.fullmatch(text) is not None or
                (include_give and any(pattern.fullmatch(text) for pattern in GIVE)))
    if eligible:
        return question_overlap(document, question, spans, records, fraction, mode="mix")
    return choose(document, question, spans, records, fraction, "entity", .25)
