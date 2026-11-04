"""Recover whitespace-only quote variants and map them to selected source."""
from contextlib import contextmanager
import json
import re

from masked_ttt.self_study import normalized, parse_exercises as original_parse


def original_source_quote(quote, document, spans, selected):
    """Match whitespace variants inside one selected block with complete boundaries."""
    parts = re.split(r"\s+", quote.strip())
    if len(parts) < 2:
        return None
    pattern = re.compile(r"\s+".join(re.escape(part) for part in parts))
    for index in selected:
        start, end = spans[index]
        chunk = document[start:end]
        for match in pattern.finditer(chunk):
            left = chunk[match.start() - 1] if match.start() else ""
            right = chunk[match.end()] if match.end() < len(chunk) else ""
            first, last = chunk[match.start()], chunk[match.end() - 1]
            if left and left.isascii() and left.isalnum() and first.isascii() and first.isalnum():
                continue
            if right and right.isascii() and right.isalnum() and last.isascii() and last.isalnum():
                continue
            if first.isdigit() and left in ("-", "+", "−"):
                continue
            if first.isdigit() and left in (".", ",") and match.start() > 1 and chunk[match.start()-2].isdigit():
                continue
            if last.isdigit() and right in (".", ",") and match.end()+1 < len(chunk) and chunk[match.end()+1].isdigit():
                continue
            if last in (".", ",") and match.end() > 1 and chunk[match.end()-2].isdigit() and right.isdigit():
                continue
            return chunk[match.start():match.end()]
    return None


def parse_with_quote_recovery(text, document, spans, selected, original_question, limit=2):
    """Retain all strict proposals and fill free slots with validated source quotes."""
    accepted, rejections = original_parse(text, document, spans, selected, original_question, limit)
    if len(accepted) >= limit or not rejections.get("quote_not_in_selected_chunk"):
        return accepted, rejections
    decoder = json.JSONDecoder()
    items = None
    for i, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            candidate = candidate.get("exercises")
        if isinstance(candidate, list):
            items = candidate
            break
    if items is None:
        return accepted, rejections
    seen = {normalized(item.question) for item in accepted}
    selected_texts = [document[spans[i][0]:spans[i][1]] for i in selected]
    rejections = dict(rejections)
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("evidence"), str):
            continue
        evidence = item["evidence"].strip()
        if not evidence or any(evidence in chunk for chunk in selected_texts):
            continue
        quote = original_source_quote(evidence, document, spans, selected)
        if quote is None:
            continue
        repaired = dict(item, evidence=quote)
        result, _ = original_parse(json.dumps({"exercises": [repaired]}, ensure_ascii=False),
                                   document, spans, selected, original_question, limit=1)
        if not result or normalized(result[0].question) in seen:
            continue
        accepted.append(result[0])
        seen.add(normalized(result[0].question))
        rejections["quote_not_in_selected_chunk"] -= 1
        if rejections["quote_not_in_selected_chunk"] <= 0:
            rejections.pop("quote_not_in_selected_chunk")
        if len(accepted) == limit:
            break
    return accepted, rejections


@contextmanager
def use_candidate(runtime):
    """Temporarily patch a single-request runtime and restore the strict parser."""
    prior = runtime.parse_exercises
    if prior is not original_parse:
        raise ValueError("The baseline self-study parser has changed")
    runtime.parse_exercises = parse_with_quote_recovery
    try:
        yield
    finally:
        runtime.parse_exercises = prior
