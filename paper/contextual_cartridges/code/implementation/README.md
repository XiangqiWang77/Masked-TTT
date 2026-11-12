# Implementation review snapshot

These files are copied from the current working implementation of Contextual
Cartridges: answer-guided query-specific dataset crafting plus ED-JSD
selective self-distillation into a frozen-backbone native KV cache. They are
not a standalone package: execution requires the full Query_Cart project and
the official Cartridges library. The copy is for review and does not alter
any running experiment. See [hashes](../../data/implementation_manifest.json).

| File | Role |
| --- | --- |
| `ed_jsd.py` | ED-JSD selective SD: warmup, source-support admission pool, forward-only coarsened-distribution gate, optional teacher-confidence condition, exact ordinary-SD fallback |
| `source_gate.py` | Deterministic structural Source-Support Gate recorded at Teacher generation (empty/truncated/refused answers) |
| `answer_guided_questions.py` | Answer-guided question crafting: conclusion extraction then closed-book question construction, with deterministic filters |
| `question_guided_candidates.py` / `qwen_reranker_candidates.py` | Reranker-based single/pair evidence-group selection |
| `early_study.py` | Shared SD training utilities (packing, dataset, evaluation of distillation) |
| `run_figure4_contextual.py` | Three-dataset cache-size campaign runner (LongHealth, MTOB KE, QASPER) |
| `run_ratio_retrain.py` / `retrain_plan.py` | Paired vanilla/contextual retraining with matched optimizer-update budgets |
| `thirty_study_data.py` / `run_thirty_study.py` | Thirty-trajectories-per-request matched synthesis (specific and agnostic arms) and its staged job runner |

Earlier gradient-influence selector files were removed from this snapshot;
the current selection rule is entirely the forward-only ED-JSD gate described
in [`ED_JSD_SD.md`](../../ED_JSD_SD.md).
