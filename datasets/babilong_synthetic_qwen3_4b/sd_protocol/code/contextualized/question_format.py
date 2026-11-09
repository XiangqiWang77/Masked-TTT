"""Conservative format checks for generated study requests, not fact checking.

These checks use neither a reference answer nor a Teacher correctness score.
They reject clear violations of the Bot A request-only contract. An accepted
request can still contain an unsupported premise; acceptance is not evidence
verification. The upstream query-agnostic generator does not use this check.
"""
import re


QUERY_REQUEST_FORMAT_VERSION = "request-only-v1"

_SOURCE_BLOCK = re.compile(
    r"<\s*/?\s*(?:context|task_context|task_rules|info|document|example)\b[^>]*>",
    re.IGNORECASE,
)
_ANSWER_LINE = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*\S", re.IGNORECASE | re.MULTILINE)
_REQUEST = re.compile(
    r"\b(?:summari[sz]e|structure|organize|organise|explain|describe|identify|"
    r"list|compare|contrast|outline|trace|reconstruct|extract|convert|format|"
    r"provide|create|generate|write|show|tell|give|determine|analy[sz]e|"
    r"discuss|evaluate|please|request|summary|timeline|overview)\b",
    re.IGNORECASE,
)
# Be deliberately narrow: imperative and nominal requests need not contain a
# question mark. Reject an unambiguous declarative response, not every unfamiliar
# sentence. In particular, do not judge whether its location/value is correct.
_DECLARATIVE = re.compile(
    r"^\s*(?:the|a|an)\s+[^.!?\n]+\s+(?:is|are|was|were)\s+[^?\n]+[.!]?\s*$",
    re.IGNORECASE,
)


def query_request_rejection(text):
    """Return a format reason for clear non-requests; otherwise return None."""
    if not text.strip():
        return "empty_request"
    if _SOURCE_BLOCK.search(text):
        return "embedded_source_or_task_block"
    if _ANSWER_LINE.search(text):
        return "supplied_answer_line"
    if "?" not in text and not _REQUEST.search(text) and _DECLARATIVE.fullmatch(text):
        return "declarative_answer_in_request_role"
    return None
