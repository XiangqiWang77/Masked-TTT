# Self-study JSON failures in completed development runs

The existing OLMo-style self-study configuration generates up to 192 tokens
and can add independently verified, selected-evidence practice questions.
This aggregate audit uses completed Qwen and GPT-OSS development logs,
200 cases per model from 50 source questions at four context lengths. It
reads saved counts and rejection flags, with no new model or API calls; model
predictions and reference answers are not used in the calculation.

| Model | At least one verified practice | JSON failures | Failures at 192-token limit | Failures below limit |
|---|---:|---:|---:|---:|
| Qwen3-4B | 40 / 200 | 62 | 2 | 60 |
| GPT-OSS-20B | 73 / 200 | 0 | 0 | 0 |

For the 60 Qwen failures below the limit, seven produced two tokens and 53
produced four. Token counts do not establish why these responses failed, and
the generated JSON text is not preserved in these records. Thus the logs do
not show how many complete objects can be recovered. The two examples at
the limit could be truncated, but a limit hit alone does not prove this.

The two model runs used the same selection family and OLMo self-study budget;
their Value learning rates differed. The runs are development data and are not
evidence of final accuracy. We keep the experimental parser disabled while
continuing the already submitted paired optimization and holdout experiments.

[Aggregate counts](summary.json) · [Audit script](../../../experiments/truncated_study_json/audit_development.py) ·
[Candidate and safety filters](../../../experiments/truncated_study_json/README.md).
