# Recover complete practice items from unfinished JSON

The original OLMo-style method optionally generates evidence-supported practice
questions and independently verifies each one. Its parser currently discards
an entire response if the surrounding JSON cannot be decoded. This isolated
candidate can recover complete JSON objects inside an `exercises` array even
when a later part of the array is incomplete; it does not invent or repair
fields. It then uses the original selected-chunk quote check, answer-in-quote
check, question deduplication and independent model verifier.

The [development log audit](../../docs/analysis/study-json-development-20261104/README.md)
found that only two recorded JSON failures in the inspected Qwen development
run reached the configured generation limit; no GPT-OSS development cases had
this failure. Generated self-study text
was not retained in these records, so they cannot directly establish whether
any complete object could be recovered. A real same-output probe would be
needed before claiming an increase in verified exercises or accuracy.
An older GPT-OSS run did retain 200 short previews; replaying those recovered
no additional structurally valid proposals. See the same audit for limitations.

This is an experimental, unsubmitted candidate. It is not enabled in the
production runtime or any currently queued or running job. Its scoped runtime
patch is intended only for a single-request paired experiment and restores the
original parser on exit. The SD objective, cache paths, selector and verifier
are unchanged. No additional model calls are introduced by the parser itself;
an extra verifier call may occur for each newly recovered exercise.

[Candidate](parser.py) · [Audit](audit_development.py) ·
[Parser tests](../../tests/test_truncated_study_json.py)
