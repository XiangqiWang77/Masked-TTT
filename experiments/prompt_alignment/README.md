# Same-question-format SD candidate

The historical OLMo-style implementation asks the selected-context Teacher to
follow a generic answer-format instruction while the full-context Student's
main question uses the original task prompt. The completed paired audit in
[prompt-format-audit-20261104](../../docs/analysis/prompt-format-audit-20261104/README.md)
shows that the generic instruction can affect answers. Whether aligning the
training question itself helps distillation has not yet been measured.

The opt-in `align_student_teacher_prompt` flag uses the Teacher's **question
suffix only** as the full-document Student's main SD question. It does not
include the Teacher's generated answer. The selected Teacher and full Student
retain independent caches; Student prefixes remain on-policy, the target remains
the same full-vocabulary Teacher-to-Student KL, and only the original selected
KV gain and Value coefficients are trained. The verified practice paths retain
their existing question construction. `standalone_answer_format` separately
selects whether the independent materialized cartridge answers with the original
question or the same generic requirement.

The default flag is false and preserves the archived OLMo-style setup and all
frozen running-job snapshots. If enabled, an overlong aligned question and
rollout fail explicitly rather than silently truncating the document or changing
the evidence. The option does not use evaluation answers or change the cached
Decision API scores.

Compare the enabled variant against the existing aligned-output control on the
same cases, selected chunks, Teacher, rollout budget, optimizer and final
question format. Report the same-cache zero-update and trained results, the
Teacher result, and per-case gains and losses. A small-model CPU test checks
the two question paths, both gradient branches, and frozen weights and KV; it
does not establish an accuracy gain.

Development job **31986468** uses one GPU to run the same frozen 200 cases for
Qwen3-4B and GPT-OSS-20B in sequence. It compares against the original 32-token,
two-base-step checkpoints, reuses cached Decision scores, and performs a
loaded-model preflight before evaluation. Reporting runs hourly; results are
not yet available. The experiment is development only, not a 1,000-case result.
