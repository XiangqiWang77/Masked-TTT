# Whitespace-only recovery of source quotes

The OLMo-style method optionally asks the model for short exercises and checks
that each proposed evidence quote occurs in a selected chunk. Model-generated
quotes can differ from the source only in spacing or line breaks. This
experimental parser can locate such a quote in **one** selected chunk, replace
it with the exact source substring and rerun the original answer-in-quote,
question and length checks. The original independent model verifier still
decides whether a recovered exercise can enter the SD training path.

The candidate preserves every strict proposal and uses only remaining exercise
slots. It never changes letters, digits or punctuation, matches across chunk
boundaries, or repairs a partial numeric boundary. It is isolated behind a
scoped parser patch and is **not enabled in any active job or public runtime**.
No labels or additional API calls are used. A newly recovered exercise can
require one additional verifier call.

[Replaying 200 older GPT-OSS practice previews](../../docs/analysis/study-quote-recovery-20261104/README.md)
found five additional structurally accepted proposals on five cases, with no
strict proposal displaced. These proposals have not been independently
verified, used for training or evaluated for answer accuracy. Current Qwen and
GPT development logs show quote-rejection events, but do not preserve their
generated text, so we do not know how many of those can be recovered.

[Candidate](parser.py) · [Saved-preview replay](audit_saved_previews.py) ·
[Safety and verifier tests](../../tests/test_whitespace_quote_recovery.py)
