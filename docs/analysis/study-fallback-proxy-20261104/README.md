# Would a second self-study prompt help the Qwen development cases?

The completed same-GPU study-prompt experiment compared the original and an
alternate prompt with the same selected evidence, Teacher and zero-update
answer. Here we examine a conditional idea: keep the original practice if it
passes verification; otherwise try the alternate prompt. No fallback was
implemented or run. The following counts are a proxy formed from the two
already generated arms on 200 development cases from 50 source questions.

| Quantity | Count |
|---|---:|
| Original verified exercises | 48 |
| Original cases with at least one verified exercise | 41 |
| Cases with no original exercise but an alternate exercise | 17 |
| Source questions represented by those 17 cases | 14 |
| Potential additional verified exercises in those cases | 20 |
| Alternate-answer new correct / new wrong in those cases | 0 / 0 |

Replacing the saved original-arm answer with the saved alternate-arm answer
only for those 17 cases leaves the proxy at 118 / 200 correct, equal to the
original arm. The same-run Teacher has 119 / 200 correct. This does not prove
the result of actually running the conditional procedure: generating a second
prompt in one request would need a separate implementation and evaluation.
The group was selected post hoc, and questions repeat across four lengths.

The alternate-prompt run recorded eight JSON failures; an earlier original
prompt run on the same development cases recorded 62. Those rejection counts
are from different runs and hardware may differ. The proxy gives no evidence
of an accuracy improvement from simply filling cases
with more verified exercises. We therefore do not submit another GPU job for
this fallback; the paired update and holdout jobs continue unchanged.

[Aggregate counts](summary.json) · [Audit script](../../../experiments/self_study_prompt/freeze_fallback_proxy.py) ·
[Full study-prompt comparison](../qwen-study-prompt-20261104/README.md).
