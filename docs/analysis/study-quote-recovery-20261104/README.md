# Candidate recovery of evidence quotes

In the completed OLMo-style development runs, the strict parser recorded 23
quote-not-in-selected-chunk rejection events for Qwen and 46 for GPT-OSS.
These are proposal-level events, not distinct cases. The logs retain counts
but not the generated text, so we cannot classify those particular rejections
or infer whether a whitespace-only repair would help.

An older GPT-OSS run retained previews of at most 512 characters. A replay of
200 cases from 50 source questions produced the following structurally valid
proposals; the candidate maps whitespace variants back to exact selected
source spans before applying the original filters.

| Parser | Accepted proposals | Additional proposals | Cases with additions |
|---|---:|---:|---:|
| Strict original | 278 | — | — |
| Whitespace-only recovery | 283 | 5 | 5 |

None of the original strict proposals were displaced. One saved preview was
512 characters, so the replay may not contain the complete generated output.
The five additions were not independently verified or trained, and no answer
accuracy was evaluated. The older run used a different selection configuration
from the current runs. This is a bounded diagnostic, not a performance result.

The candidate is off by default and has not been submitted as a job. A future
same-GPU comparison would need to check verifier acceptance and compare the
final independent Cartridge with the same selected Teacher and zero-update
controls before considering a source-disjoint holdout.

[Aggregate replay](summary.json) · [Replay code](../../../experiments/whitespace_quote_recovery/audit_saved_previews.py) ·
[Current rejection counts](../study-json-development-20261104/summary.json) ·
[Candidate](../../../experiments/whitespace_quote_recovery/README.md)
