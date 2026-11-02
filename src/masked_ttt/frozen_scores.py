"""CPU-produced source scores; GPU readers have no network fallback."""

from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import time

from .request import Layout, text_sha
from .selection import PreparedRequest, compact_source, select_top_fraction


SCHEMA = "masked-ttt-source-scores-v1"


def score_key(document, question):
    return hashlib.sha256(
        json.dumps([SCHEMA, text_sha(document), question], ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_scores(directory, document, question):
    path = Path(directory) / "scores" / (score_key(document, question) + ".json")
    data = json.loads(path.read_text())
    if (
        data.get("schema") != SCHEMA
        or data.get("complete") is not True
        or data.get("document_sha256") != text_sha(document)
        or data.get("question_sha256") != text_sha(question)
        or data.get("reference_answers_used") is not False
    ):
        raise ValueError("Frozen source score identity/format mismatch")
    spans = data["source_spans"]
    if len(spans) != len(data["records"]) or len(spans) != len(data["chunk_sha256"]):
        raise ValueError("Incomplete frozen chunk scores")
    previous = 0
    for (start, end), digest in zip(spans, data["chunk_sha256"]):
        if not 0 <= start < end <= len(document) or start < previous:
            raise ValueError("Invalid or overlapping frozen source ranges")
        if text_sha(document[start:end]) != digest:
            raise ValueError("The frozen source passage changed")
        previous = end
    select_top_fraction(data["records"], 0.1)
    data["json_path"] = str(path)
    data["json_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def layout_on_source_spans(backbone, request, spans):
    """Map shared character chunks to each model's disjoint native token slots."""
    text = (
        backbone.tokenizer.apply_chat_template(
            [dict(role="user", content=request.prompt)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + request.answer_prefix
    )
    begin = text.find(request.document)
    if begin < 0 or text.find(request.document, begin + 1) >= 0:
        raise ValueError("The source must occur once in the original prompt")
    encoded = backbone.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = [s[1] for s in spans]
    chunks = [[] for _ in spans]
    initial = None
    for index, (a, b) in enumerate(encoded["offset_mapping"]):
        if b <= begin or a >= begin + len(request.document) or a == b:
            continue
        if initial is None:
            initial = index
        point = max(0, a - begin)
        bank = min(bisect_right(ends, point), len(spans) - 1)
        chunks[bank].append(index)
    if initial is None or not initial or any(not c for c in chunks):
        raise ValueError("A canonical source chunk has no native token representation")
    if chunks[-1][-1] + 1 >= len(encoded["input_ids"]):
        raise ValueError("A question suffix is required")
    return Layout(encoded["input_ids"], chunks, initial, [tuple(x) for x in spans])


class FrozenJSONSelector:
    """Consume exactly one immutable source-score file. Never constructs an API client."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def prepare(self, backbone, original):
        began = time.perf_counter()
        data = load_scores(self.directory, original.document, original.question)
        spans = data["source_spans"]
        layout = layout_on_source_spans(backbone, original, spans)
        original_tokens = len(layout.ids)
        records = data["records"]
        working, mapping = original, None
        limit = backbone.config.max_context - max(
            backbone.config.context_reserve_tokens, original.max_new_tokens
        )
        if original_tokens > limit:
            source_layout, source_records = layout, records
            budget = limit - (len(layout.ids) - sum(map(len, layout.chunks))) - 1024
            for _ in range(8):
                document, mapping = compact_source(
                    original.document, spans, source_records, source_layout.chunks, budget
                )
                working = original.with_document(document)
                retained_spans = [(r["start"], r["end"]) for r in mapping]
                layout = layout_on_source_spans(backbone, working, retained_spans)
                if len(layout.ids) <= limit:
                    records = [
                        dict(
                            source_records[r["original_chunk"]],
                            chunk=i,
                            original_chunks=[r["original_chunk"]],
                        )
                        for i, r in enumerate(mapping)
                    ]
                    break
                budget -= len(layout.ids) - limit + 1024
            else:
                raise ValueError("The source and original task exceed the model window")
        selected = select_top_fraction(records, backbone.config.keep_fraction)
        return PreparedRequest(
            working,
            layout,
            selected,
            dict(
                backend="precomputed_source_JSON",
                score_file=data["json_path"],
                score_file_sha256=data["json_sha256"],
                canonical_tokenizer=data["canonical_tokenizer"],
                selected_chunks=selected,
                selected_original_chunks=[records[i].get("original_chunks", [i]) for i in selected],
                total_chunks=len(layout.chunks),
                selected_tokens=sum(len(layout.chunks[i]) for i in selected),
                original_prompt_tokens=original_tokens,
                working_prompt_tokens=len(layout.ids),
                complete_original_document=mapping is None,
                source_mapping=mapping,
                source_sha256=text_sha(original.document),
                working_source_sha256=text_sha(working.document),
                reference_answers_used=False,
                online_api_calls=0,
                offline_scoring_seconds=data["scoring_seconds"],
                offline_scoring_cost_usd=data["estimated_cost_usd"],
                seconds=time.perf_counter() - began,
            ),
        )
