# Contextual Cartridges: evaluation protocol

Status: 2026-11-09. The synthetic datasets are complete. The corrected native-KV
learning comparison is in development; an accuracy improvement has not yet
been established for both components.

Both Qwen3-4B datasets cover the same 1,000 BABILong cases: ten task types,
four lengths (8K, 16K, 32K, 64K), and 25 cases per task-length cell. Each has
20,558 training and 7,391 validation conversations, with identical split sizes
within every case. Existing questions are reused rather than regenerated.

The four comparisons are:

| Synthetic questions | Learning rule | Purpose |
|---|---|---|
| Original query-agnostic | Vanilla SD | Cartridges control |
| Original query-agnostic | Selective SD | Isolate selective learning |
| Query-specific, answer-guided | Vanilla SD | Isolate question construction |
| Query-specific, answer-guided | Selective SD | Complete Contextual Cartridges |

The original control uses the released SelfStudySynthesizer question/source
bindings. Its Teacher reads the actual sampled source and uses Bot B's
temperature 0, with real top-20 soft targets. This is an implementation-based
BABILong control, not a claimed reproduction of a numerical result from a
different dataset in the Cartridges paper.

All four arms start from the same per-case 512-token native KV memory, freeze
the backbone, use the original sparse distillation objective and packing, and
match actual executed optimizer steps. Early stopping executes at least 64
steps and retains the best *already updated* validation checkpoint, including
checkpoints before step 64. The other arms execute the number of steps used by
the query-agnostic Vanilla-SD control and select checkpoints by the same rule.
Neither benchmark gold answers nor answer-first construction anchors are
Student training labels.

Selective SD leaves Teacher trajectories and token distributions unchanged.
It removes a trajectory only when the source-grounded contradiction rule
vetoes it. An unresolved or truncated judgment retains ordinary SD. If all
trajectories are rejected, the request falls back to its original SD data;
unreliable verification cannot make the Cartridge skip training.

The earlier full-document/temperature-0.6 Teacher study, the strict
support-required study, and the minimum-step-only checkpoint study are kept
as separate historical runs. Their results are not pooled with this protocol.
The older reported Selective-SD gain came from a different cache
parameterization and does not establish a gain for this native-KV experiment.

## What each result can establish

The published synthetic-data audit measures source exposure. No-support
material fell from 83.78% to 46.69%, and support-union recall rose from 52.31%
to 91.06% on 993 consistently mapped cases. In the 296 cases whose support
crosses original chunk boundaries, the fraction of exercises exposed to all
support rose from 0% to 23.66%. These are useful dilution and fragmentation
diagnostics, not answer accuracy, proof of necessary multi-hop reasoning, or
proof that all originally useful material was retained. The full report keeps
regressions, exclusions, and paired source-cluster uncertainty.

The question-construction contribution is measured by query-specific versus
query-agnostic questions with Vanilla SD fixed. The selective-learning
contribution is measured with the synthetic data fixed, separately on both
datasets. We report paired fixes and regressions, not only favorable cases.

Beyond-Teacher analysis includes full-context-Teacher-wrong/Student-correct
cases and the reverse. Stronger evidence for selective learning additionally
requires that the initial memory and the Vanilla-SD arm were wrong, while the
updated Selective-SD arm became correct. A correct untrained prior is not
counted as a newly learned correction. Full-context accuracy is a comparator,
not a theoretical upper bound and not the source-conditioned training target.

## Development and independent confirmation

The existing source-disjoint split contains 160 development cases (40 source
questions) and 840 remaining cases (210 source questions). Earlier SD runs
had already evaluated 127 of those remaining cases. Marking every document
length of their source questions as previously evaluated leaves:

| Evaluation partition | Cases | Source questions |
|---|---:|---:|
| Development | 160 | 40 |
| Remaining, source observed in prior SD runs | 388 | 97 |
| Remaining, source not observed in the listed prior SD runs | 452 | 113 |

This provenance is frozen without using the prior accuracy values. Training
inputs and the original 160/840 assignment do not change. The 452-case subset
is reported separately for independent confirmation; all 840 are still
reported, with their prior evaluation disclosed. No outcome from the 452-case
confirmation subset is used to select hyperparameters or admission rules.

Accuracy uses the unchanged official BABILong evaluator. Comparisons share
the same completed cases. Confidence intervals resample source questions
within each task, retaining their document lengths together. Missing runs are
pending, never silently counted as incorrect. Partial results are exploratory;
individual 95% intervals are not simultaneous intervals for all claims.

## Resources and reproducibility

The corrected run reuses the existing GPU job arrays. It adds no GPU jobs,
allows at most 12 jobs/GPUs, excludes A40, and keeps the original campaign
deadline of 2026-11-09 10:51 UTC. Pre-handoff checks use static code and saved
CPU artifacts; no GPU smoke test is run. Source targets, admission records,
and initial memories are reused, while fitted memories are recomputed after
the checkpoint correction. Monitoring refreshes artifacts every five minutes
and queries the scheduler hourly.

The frozen protocol, provenance manifest, per-case update counts, fallback
counts, fitted memories, predictions, and correction/regression lists provide
the audit trail. Improvement in validation distillation loss alone does not
prove that either component improves final answers.
