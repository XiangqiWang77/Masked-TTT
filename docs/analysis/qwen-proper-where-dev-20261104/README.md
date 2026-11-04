# Qwen: question-guided evidence selection and Cartridge

This completed 200-case development comparison tests the existing Qwen Query
Cartridge with a question-guided selector. Both selectors use the same cached
Decision scores and the same chunk budget; the candidate supplements the
baseline only for a simple name-location question. The candidate was designed
after inspecting development coverage. Selection and generation do not use
reference answers; results use `answer_focused_lexical_v2` after generation.

| Selection | Whole | ICL Teacher | Selected-cache zero update | Cartridge |
|---|---:|---:|---:|---:|
| Existing entity-linked | 58.50% | 59.50% | 57.50% | 58.50% |
| Question-guided candidate | 58.00% | **61.50%** | **59.50%** | **60.00%** |

The candidate Cartridge improves by **1.50 percentage points** over the saved
original Cartridge (4 newly correct, 1 newly wrong). The two runs may use
different GPUs. Although the evidence selection is identical for 193 cases,
one of those cases changes from correct to incorrect; 189 of the 193 have
identical Cartridge output tokens. The results therefore include cross-run
drift as well as the intended selection change.

The selection changes for seven cases, all `qa1` and originating from four
different source questions. Among these seven, both the Teacher and Cartridge
correct four additional cases relative to their saved counterparts, with no
new errors; those four gains cover four source questions. The zero-update and
trained Cartridge have the same correctness on all seven, so the changed-set
gain here is evidence about selection, not evidence that SD caused the gain.
The Teacher and Cartridge use different generic answer-format instructions.

A post-hoc annotated-span check finds that six of the seven changed cases move
from incomplete to complete support coverage; the other already had complete
coverage. All four new correct cases for the Teacher, zero-update cache and
Cartridge are among the six newly covered cases. This connects the improvement
to recovered evidence in these examples, though coverage alone does not prove
which tokens the model used. The selector did not receive the annotations.
[Coverage transition audit](coverage_transition.json) · [reproduction script](../../../experiments/babilong_linked_selection/audit_proper_where_coverage.py).

The candidate Cartridge remains **1.50 points below its own Teacher** (1 newly
correct, 4 newly wrong). It did not pass the predeclared rule requiring it to
beat all controls, so the 800-case evaluation was not launched. All 200 cases
come from 50 source questions at four context lengths; the candidate was
tuned on these development observations, not verified on a disjoint holdout.

The corresponding [coverage audit](../question-overlap-coverage-dev-20261104/README.md)
measured supporting-fact coverage rather than answer accuracy. The
[aggregate-only snapshot](summary.json) and [freezing script](../../../experiments/babilong_linked_selection/freeze_proper_where.py)
use saved predictions; no new model or API calls were made.
