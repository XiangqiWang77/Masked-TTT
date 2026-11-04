# Same-GPU gain learning-rate pair

The completed Qwen readout probe found that the first-token distribution could
change in the full context while changing much less in the independent selected
cache. A stronger per-chunk gain update is a testable candidate, not a guarantee
of transfer or accuracy. The existing 200-case Value-rate sweep kept gain at
0.01; this experiment tests 0.03 against 0.01 on the same request and GPU.

Each model uses its own fixed Value rate in both arms: 0.0001 for Qwen3-4B and
0.001 for GPT-OSS-20B. Both arms independently prefill the complete document
Student and selected-evidence Teacher, use the same cached Decision scores,
10% linked chunk selection, on-policy forward KL, 8-token Student rollout,
two base updates and verified optional practice. Only `gain_learning_rate`
changes within a model. The learned per-chunk gains and low-rank Value state are
materialized onto an independently prefilled selected cache; full and source KV
and training state are released before final generation. No evaluation answer
is used for selection, training or inference; there are no online API calls.

One six-hour, requeue-capable single-GPU job evaluates the fixed 200-case
development partition for each model, sharing one loaded model and device for
each pair. The runner verifies identical selected evidence, Teacher output
tokens, independently verified practice, native answer and zero-update answer.
It does not compare the full Teacher probability tensors. The hourly reporter
grades all arms with the same `answer_focused_lexical_v2` loose lexical metric.
The Teacher has its original generic formatting instruction; both cartridge
arms and the selected-zero control use the original question format.

Each model is promoted independently to a 200-source, 800-case holdout only
when all 200 valid development pairs beat that model's whole, zero, Teacher
and 0.01 gain controls. Reports include paired new correct/new wrong counts,
per-task and per-length results, and standalone KV size. Source questions do
not cross the development/holdout split. Two holdout source questions and
aggregate results under other configurations have previously been examined;
this is a source-disjoint holdout, not a fully blind evaluation. The goal is
to replicate the OLMo improvement; actual performance is determined by the
held-out results.

[Runner](runner.py) · [Reporter](report.py) · [Submission](job.py) ·
[Hourly monitor and promotion](monitor.py) · [Tests](../../tests/test_gain_rate_pair.py)
