"""Preserve the original question, task formatting, and source token positions."""

from dataclasses import dataclass
from ._layout import chunk_positions


@dataclass(frozen=True)
class Request:
    document: str
    question: str
    prompt: str | None = None
    answer_prefix: str = ""
    max_new_tokens: int = 1024

    def __post_init__(self):
        if not isinstance(self.document, str) or not self.document.strip():
            raise ValueError("A nonempty source document is required")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("A nonempty original question is required")
        if self.prompt is None:
            object.__setattr__(
                self,
                "prompt",
                "Read the following context and answer the question.\n\n"
                + self.document
                + "\n\nQuestion: "
                + self.question
                + "\nAnswer concisely.",
            )
        if self.prompt.count(self.document) != 1 or self.question not in self.prompt:
            raise ValueError("The prompt must contain the source once and the original question")
        if not isinstance(self.answer_prefix, str):
            raise ValueError("answer_prefix must be text")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

    def with_document(self, document):
        return Request(
            document=document,
            question=self.question,
            prompt=self.prompt.replace(self.document, document, 1),
            answer_prefix=self.answer_prefix,
            max_new_tokens=self.max_new_tokens,
        )


@dataclass
class Layout:
    ids: list
    chunks: list
    initial: int
    spans: list

    @property
    def document_end(self):
        return self.chunks[-1][-1] + 1


def prepare_layout(backbone, request):
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
        raise ValueError("Source document must occur exactly once in the chat template")
    end = begin + len(request.document)
    encoded = backbone.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    chunks = chunk_positions(offsets, begin, end, text, backbone.config.chunk_tokens, "legacy")
    if not chunks:
        raise ValueError("The source has no complete token chunk")
    initial = next(i for i, (a, b) in enumerate(offsets) if b > begin and a < b)
    if initial < 1 or chunks[-1][-1] + 1 >= len(encoded["input_ids"]):
        raise ValueError("An original task prefix and a question suffix are required")
    spans = [
        (max(begin, offsets[c[0]][0]) - begin, min(end, offsets[c[-1]][1]) - begin) for c in chunks
    ]
    return Layout(encoded["input_ids"], chunks, initial, spans)
