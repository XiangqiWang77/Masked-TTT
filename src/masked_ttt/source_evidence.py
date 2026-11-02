"""A diagnostic critic: retrieve without the candidate, then verify its signal."""

from dataclasses import dataclass
import hashlib
import json


EXTRACTION = """The original source and task are above. Retrieve the exact source
passages needed to assess the original question, repeated below. Do NOT answer
the question. No proposed answer is provided. Include the necessary entities,
relations, intermediate facts and relevant later updates. For a tracked item,
include the carrier's movements and transfers; resolve "there" using source
events. For a spatial relation include the links needed to establish direction.
Return one JSON object: {"evidence":["short verbatim source passage", ...]}.
Use at most12 short source quotes. Never invent, paraphrase or repair a quote.
"""

VERIFICATION = """Assess this unchanged candidate against the exact original
question and the verbatim source excerpts below. The candidate is untrusted.
The numeric positions specify the original document order, not answer values.
First interpret the question and candidate as a factual proposition, preserving
the argument order in relations and the precise before/after condition. Then
check that proposition. Track possession and location through the quoted
events. A state persists until a relevant event changes it; dropping an item
leaves it where it was dropped. Multi-step reasoning and equivalent wording
are allowed. The excerpts can be incomplete: an absence of explicit wording
is not a contradiction, and an earlier state alone does not prove the latest.
Do not rewrite the candidate or produce a replacement answer.
Return one JSON object with keys in this order:
{"reason":"concise source-based assessment of the candidate itself",
 "evidence":["short exact source quote"],"candidate_has_answer":true,
 "candidate_supported":false,"candidate_contradicted":false}.
The booleans are schema examples. supported means the source facts establish
THIS candidate answer to THIS question; contradicted means source facts
establish an incompatible answer. They cannot both be true. If uncertain,
make both false. An unambiguous candidate answer need not include reasoning.
"""


@dataclass(frozen=True)
class EvidenceJudgeOptions:
    extraction_tokens: int = 512
    verification_tokens: int = 384
    maximum_quotes: int = 12
    maximum_occurrences: int = 64

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in self.__dict__.values()):
            raise ValueError("Require positive integer evidence-critic budgets")


def parse_object(text):
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def verified_occurrences(quotes, document, maximum=64):
    source = " ".join(document.split())
    matches = {}
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            continue
        normalized = " ".join(quote.split())
        if len(normalized) > 600 or len(normalized.split()) < 3:
            continue
        offset = 0
        while True:
            start = source.find(normalized, offset)
            if start < 0:
                break
            matches[(start, start + len(normalized), normalized)] = dict(
                start=start, end=start + len(normalized), quote=normalized
            )
            if len(matches) > maximum:
                return [], "too_many_source_occurrences"
            offset = start + 1
    return [matches[k] for k in sorted(matches)], None


def parse_verification(text, document, *, complete=True):
    fallback = dict(
        resolved=False,
        apply_update=None,
        parse_valid=False,
        reason="unresolved_independent_verification",
        evidence=[],
        source_quotes_verified=False,
    )
    value = parse_object(text) if complete else None
    keys = [
        "reason",
        "evidence",
        "candidate_has_answer",
        "candidate_supported",
        "candidate_contradicted",
    ]
    if (
        not isinstance(value, dict)
        or list(value) != keys
        or not isinstance(value["reason"], str)
        or not value["reason"].strip()
        or not isinstance(value["evidence"], list)
        or any(not isinstance(q, str) or not q.strip() for q in value["evidence"])
        or any(type(value[k]) is not bool for k in keys[2:])
        or value["candidate_supported"]
        and value["candidate_contradicted"]
    ):
        return fallback
    source = " ".join(document.split())
    grounded = bool(value["evidence"]) and all(
        " ".join(q.split()) in source for q in value["evidence"]
    )
    supported = bool(value["candidate_supported"] and grounded and value["candidate_has_answer"])
    rejected = bool(
        not value["candidate_has_answer"] or value["candidate_contradicted"] and grounded
    )
    return dict(
        value,
        resolved=supported or rejected,
        apply_update=(not rejected if supported or rejected else None),
        parse_valid=True,
        source_quotes_verified=grounded,
    )


def independent_evidence_judge(
    engine,
    reader,
    cache,
    ids,
    document_length,
    prompt,
    document,
    question,
    candidate,
    old_judgment,
    options,
    *,
    skip_generic_verification=False,
):
    import torch
    from transformers.cache_utils import DynamicCache
    from .cache import fork_prefix
    from .support_views import _generate_assessment

    stamps = [
        (x, x._version, x.data_ptr()) for layer in cache.layers for x in (layer.keys, layer.values)
    ]
    content = (
        prompt
        + "\n\n"
        + EXTRACTION
        + "\nOriginal question (JSON):\n"
        + json.dumps(question, ensure_ascii=False)
    )
    rendered = engine.tokenizer.apply_chat_template(
        [dict(role="user", content=content)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = engine.tokenizer.encode(rendered, add_special_tokens=False)
    if tokens[:document_length] != ids[:document_length].tolist():
        raise ValueError("Evidence retrieval changed the original document prefix")
    with torch.no_grad():
        extracted = _generate_assessment(
            engine,
            reader,
            fork_prefix(cache, document_length),
            tokens[document_length:],
            document_length,
            options.extraction_tokens,
        )
        data = parse_object(extracted["prediction"]) if extracted["complete"] else None
        valid = (
            isinstance(data, dict)
            and list(data) == ["evidence"]
            and isinstance(data["evidence"], list)
            and len(data["evidence"]) <= options.maximum_quotes
            and all(isinstance(q, str) and q.strip() for q in data["evidence"])
        )
        quotes = list(data["evidence"]) if valid else []
        # Keep only source text from the old critics, never their conclusions.
        for assessment in (old_judgment, old_judgment.get("focused_assessment") or {}):
            quotes.extend(assessment.get("evidence", []))
        occurrences, ambiguity = verified_occurrences(quotes, document, options.maximum_occurrences)
        verified = dict(
            resolved=False,
            apply_update=None,
            parse_valid=False,
            reason=ambiguity or "no_source_verified_evidence",
            evidence=[],
            source_quotes_verified=False,
        )
        generation = None
        if occurrences and not skip_generic_verification:
            context = (
                "Verbatim source excerpts, in original document order:\n"
                + json.dumps(occurrences, ensure_ascii=False)
                + "\n\n"
                + VERIFICATION
                + "\nOriginal question (JSON):\n"
                + json.dumps(question, ensure_ascii=False)
                + "\nUnchanged candidate (JSON):\n"
                + json.dumps(candidate, ensure_ascii=False)
            )
            text = engine.tokenizer.apply_chat_template(
                [dict(role="user", content=context)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = engine.tokenizer.encode(text, add_special_tokens=False)
            generation = _generate_assessment(
                engine,
                reader,
                DynamicCache(config=engine.model.config),
                inputs,
                0,
                options.verification_tokens,
            )
            verified = parse_verification(
                generation["prediction"], document, complete=generation["complete"]
            )
    if any(x._version != version or x.data_ptr() != pointer for x, version, pointer in stamps):
        raise AssertionError("The independent critic modified original document KV")
    return dict(
        verified,
        extraction=extracted,
        extraction_parse_valid=valid,
        extraction_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        independent_extraction_had_no_candidate=True,
        verified_source_occurrences=occurrences,
        verification=generation,
        generic_verification_skipped=skip_generic_verification,
        old_verdict_or_reason_supplied_to_model=False,
        old_decision=old_judgment["apply_update"],
        proposed_decision=verified["apply_update"]
        if verified["resolved"]
        else old_judgment["apply_update"],
        decision_rule="source-resolved independent verification; otherwise preserve original pre-update decision",
        teacher_rewritten=False,
        reference_answers_used=False,
        original_document_KV_immutable=True,
    )
