"""Recover complete exercises from a truncated JSON array, without repairing text."""
from contextlib import contextmanager
import json
import re

from masked_ttt.self_study import parse_exercises as original_parse


ARRAY_START = re.compile(r'\{\s*"exercises"\s*:\s*\[')


def complete_items(text):
    """Return only fully decoded leading objects inside an exercises array.

    A partial object, fabricated closing quote, or non-object array element is
    never recovered. The caller must still apply the original evidence checks
    and the independent model verifier.
    """
    match = ARRAY_START.search(text)
    if not match:
        return []
    decoder = json.JSONDecoder()
    cursor = match.end()
    found = []
    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            break
        try:
            value, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        found.append(value)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != ",":
            break
        cursor += 1
    return found


def parse_with_truncated_recovery(text, document, spans, selected, original_question, limit=2):
    """Preserve the original parser unless its top-level JSON is incomplete."""
    result = original_parse(text, document, spans, selected, original_question, limit)
    if result[1] != {"invalid_json": 1}:
        return result
    items = complete_items(text)
    if not items:
        return result
    # Reuse exactly the production quote, answer, length and deduplication
    # filters; independently generated quote verification remains unchanged.
    recovered = json.dumps({"exercises": items}, ensure_ascii=False)
    return original_parse(recovered, document, spans, selected, original_question, limit)


@contextmanager
def use_candidate(runtime):
    """Temporarily patch one isolated runtime; always restore its parser."""
    prior = runtime.parse_exercises
    if prior is not original_parse:
        raise ValueError("The baseline self-study parser has changed")
    runtime.parse_exercises = parse_with_truncated_recovery
    try:
        yield
    finally:
        runtime.parse_exercises = prior
