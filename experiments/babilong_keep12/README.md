# BABILONG: selection-budget development probe

This experiment holds the OLMo-style Query Cartridge method fixed and changes
the saved Decision-score selection budget from 10% to 12%. The Teacher reads
independently prefetched selected evidence, the Student reads the full document
and learns selected-position attention and Value corrections through on-policy
SD, and the standalone cartridge answers after full KV is released. There is
no additional full-to-selected loss, online API call, or change to base weights.

The 12% setting was motivated by a support-coverage sweep on the 200 development
examples: complete annotated support rose from 125 to 133 examples. Annotations
were used to evaluate selections after they were made, not as selector input.
Support coverage is not answer accuracy. The full sweep and its limitations are
described [here](../../docs/selection_coverage_development.zh-CN.md).

The public case IDs and source-disjoint split are the same as the
[Value learning-rate experiment](../babilong_value_lr/split.json): 200 cases
from 50 source questions for development and 800 cases from 200 different source
questions reserved for holdout. Each model's development report compares full
KV, 12% selected KV without updating, 12% ICL Teacher, and 12% Cartridge.
It also pairs the 12% Cartridge with the existing same-question 10% Cartridge
where that result exists. The two budgets may run on different GPU allocations;
the report records whether their full-context outputs match.

The selected Teacher has an additional generic answer-format instruction and
a shorter generation limit; the zero-update and trained selected-cache paths
use the same original question suffix and generation budget. All models use
the same task-aware loose lexical scorer, after generation. Report both new
correct and new wrong answers, with the artifact and actual runtime memory
scopes kept distinct.

A source-disjoint 800-case test is useful only after the development comparison
has selected a fixed configuration. The goal is for the independent Cartridge to
beat the Teacher, full KV, and zero-update selected KV; success remains an
empirical result. This development probe has been submitted as one single-GPU
job; it is not a completed 1,000-case result.
