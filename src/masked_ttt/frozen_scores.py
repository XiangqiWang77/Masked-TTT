"""CPU-produced source scores; GPU readers have no network fallback."""

from bisect import bisect_right
import json
from pathlib import Path
import time

from .request import Layout
from .selection import PreparedRequest, compact_source, select_top_fraction


SCHEMA = "masked-ttt-source-scores-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_scores(directory, document, question, case_key, by_case):
    path = Path(directory) / "scores" / by_case[case_key]
    data = json.loads(path.read_text())
    if (
        data.get("schema") != SCHEMA
        or data.get("complete") is not True
        or data.get("case_key") != case_key
        or data.get("question") != question
        or data.get("reference_answers_used") is not False
    ):
        raise ValueError("Frozen source score identity/format mismatch")
    spans = data["source_spans"]
    if len(spans) != len(data["records"]):
        raise ValueError("Incomplete frozen chunk scores")
    previous_start, previous_end = -1, 0
    for start, end in spans:
        if not 0 <= start < end <= len(document) or start <= previous_start or end <= previous_end:
            raise ValueError("Invalid or nonmonotone frozen source ranges")
        if start < previous_end:
            # Byte-fallback tokens can straddle a chunk boundary inside one
            # Unicode character. Their disjoint token slots legitimately map
            # to the same character offset. Preserve the scored text exactly.
            if previous_end - start != 1 or len(document[start].encode("utf-8")) <= 1:
                raise ValueError("Invalid overlapping frozen source ranges")
        previous_start, previous_end = start, end
    select_top_fraction(data["records"], 0.1)
    data["json_path"] = str(path)
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


class IndexedSource:
    """Check cached cases against the exact original document, without hashing."""

    def __init__(self, index_path, source_cases):
        self.source_cases = Path(source_cases)
        records = json.loads(Path(index_path).read_text())
        if not isinstance(records, list):
            raise ValueError("Source index must be a JSON list")
        self.entries = {}
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Each source index entry must be an object")
            key, offset, size = row.get("key"), row.get("offset"), row.get("bytes")
            if (not isinstance(key, str) or not key or key in self.entries or
                type(offset) is not int or offset < 0 or type(size) is not int or size <= 0):
                raise ValueError("Invalid or duplicate source index entry")
            self.entries[key] = (offset, size)

    def verify(self, case_key, document, question):
        if case_key not in self.entries:
            raise KeyError("No indexed source for " + case_key)
        offset, size = self.entries[case_key]
        with self.source_cases.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("The indexed source record is incomplete")
        case = json.loads(raw)
        if not isinstance(case, dict) or set(case) & {"answer", "answers", "references", "gold", "answer_spans"}:
            raise ValueError("Source cases must not contain evaluation answers")
        source_question = case.get("query", case.get("question"))
        if (case.get("key") != case_key or case.get("document") != document or
            source_question != question):
            raise ValueError("The cached case does not match the original document and question")


class FrozenJSONSelector:
    """Consume exactly one immutable source-score file. Never constructs an API client."""

    def __init__(self, directory, *, source_index=None, source_cases=None):
        self.directory = Path(directory)
        self.by_case = json.loads((self.directory/'case_to_score_file.json').read_text())
        self.case_key = None
        if (source_index is None) != (source_cases is None):
            raise ValueError("Source index and source JSONL must be supplied together")
        self.indexed_source = (IndexedSource(source_index, source_cases)
                               if source_index is not None else None)

    def bind_case(self,case_key):
        if case_key not in self.by_case:raise KeyError('No cached Decision score for '+case_key)
        self.case_key=case_key

    def prepare(self, backbone, original):
        began = time.perf_counter()
        if self.case_key is None:raise ValueError('Bind a case ID before reusing cached scores')
        if self.indexed_source is not None:
            self.indexed_source.verify(self.case_key, original.document, original.question)
        data = load_scores(self.directory, original.document, original.question,self.case_key,self.by_case)
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
                canonical_tokenizer=data["canonical_tokenizer"],
                selected_chunks=selected,
                selected_original_chunks=[records[i].get("original_chunks", [i]) for i in selected],
                total_chunks=len(layout.chunks),
                selected_tokens=sum(len(layout.chunks[i]) for i in selected),
                original_prompt_tokens=original_tokens,
                working_prompt_tokens=len(layout.ids),
                complete_original_document=mapping is None,
                source_mapping=mapping,
                source_case_key=self.case_key,
                source_document_matches_index=self.indexed_source is not None,
                working_source_length=len(working.document),
                reference_answers_used=False,
                online_api_calls=0,
                offline_scoring_seconds=data["scoring_seconds"],
                offline_scoring_cost_usd=data["estimated_cost_usd"],
                seconds=time.perf_counter() - began,
            ),
        )
