# Query Cartridges: Learning to Read Long Contexts with Masked Test-Time Training

Query Cartridges adapts a frozen language model to one long-context question.
Question-conditioned Decision scores select a small number of document chunks.
A frozen, selected-context ICL teacher supplies on-policy next-token
distributions and proposes short source-grounded practice questions. The
full-context student jointly adapts two request-local reading corrections:
chunk-level attention allocation and a low-rank value-read adjustment. Model
weights and the original document KV remain immutable.

The trained state supports two explicitly different answer paths:

* **Full KV + Query Cartridge:** answer from the working-document cache with
  the trained corrections. The default full-context judge compares native and
  adapted answers in both orders and retains the native answer if inconclusive.
  The recorded Qwen V5 profile instead selects the adapted full-context answer;
  its pre-update Teacher check remains a separate operation.
* **Standalone Query Cartridge:** answer without reading the full-document KV.
  It contains an independently prefetched selected-chunk KV together with the
  trained corrections, materialized for decoding. Its answer is evaluated
  separately from the full-context path.

The standalone cartridge is a real selected-KV cache, not just a small set of
parameters. It is constructed after full-context adaptation; consequently its
smaller decoding state does not by itself establish a lower training-time peak
memory requirement. Unlike the reusable virtual caches in
[Cartridges](https://arxiv.org/abs/2506.06266), these states are constructed for
one request and discarded afterwards.

An optional **selected-cache training** path optimizes the independently
prefilled selected KV directly and generates from that same cache. It shares
one request-local update across the original question and any verified
practice questions. The full document is still prefetched for the existing
Teacher judgment, while the final cartridge does not read that full KV.
`matched_cache_probe=True` runs an additional paired study alongside the
original full-context path; `matched_cache_direct=True` runs only the
selected-cache update and returns its independent answer as `prediction`.
The GPT-OSS evaluation of this candidate is in progress.

## Procedure

1. Load the saved Decision API scores for the document and question. Select
   the top 10% of source chunks in their original order.
2. Prefill the complete working document once. Separately prefill the selected
   text for the ICL teacher; obtain its answer for the baseline comparison.
   The Qwen BABILONG V5 configuration also checks the Teacher against the
   working document before admitting its main-question signal. A rejected
   main signal can still leave independently verified practice paths active.
3. Ask the teacher for up to two short factual practice questions, answers and
   exact source quotations. Keep only proposals whose quotation occurs in a
   selected chunk and whose answer agrees with a separate frozen-model answer
   based on that quotation alone. This is a consistency check, not a guarantee
   of logical entailment; evaluation references are never used here.
4. Generate short current-student prefixes for the original and verified
   practice questions. Score each identical prefix with the focused teacher and
   minimize full-vocabulary forward KL on the student path. All exercises share
   the same selected-position attention and value-read state.
5. Discard training suffixes and independently generate the original answer.
   Under the default full-context policy, a judge reads the working document
   and can retain the native answer. Also generate an
   independent answer with the standalone selected-KV cartridge.

For tokenizers that merge the last document character with following
whitespace, `TTTConfig(practice_boundary_fallback=True)` retries a verified
practice question with the original prompt's separator if its document prefix
would otherwise change. The retry must still match both the full-document and
selected-document cache prefixes. The option defaults to `False` to preserve
earlier experiment configurations; the GPT-OSS boundary-fix run enables it.

At a selected document location $u$ in chunk $c$, the adapted read is

$$z_{t,u}=q_t^\top k_{u,0}/\sqrt{d_h}+m_u g_c,\qquad
o_t=\sum_u \operatorname{softmax}(z_t)_u\left(v_{u,0}+
m_u\operatorname{clip}_{0.15\lVert v_{u,0}\rVert}((v_{u,0}U)B_c)\right).$$

The mask $m_u$ limits trainable changes to selected document positions; the
fixed basis $U$ is computed from their cached values. Full-attention layers
are adapted and sliding-attention layers retain their native behavior.

## Memory accounting

Each result records the backing storage of the full document KV, the
request-local adapter, optimizer state, and the materialized decode cache.
Standalone cartridge size includes both selected KV and its adapter. Shared
tensor views are counted once. A GPU peak records the full paired experiment
and includes model weights and training; it is not presented as a separately
measured standalone peak. Source and question/output suffix KV are identified
separately so that the compared cache scope is clear.

## Run

Install with `pip install -e '.[test]'` and a compatible FlashAttention build.
The saved score directory contains `case_to_score_file.json` and `scores/`.
The [shared source-score archive](https://github.com/XiangqiWang77/AsyncStack-TTT/tree/data/shared-source-scores-20261102)
contains cached scores for the 1,000 BABILONG cases and the retained
LongBench-v2 cases; the original dataset documents are still required. The
Qwen V5 profile selects its 30% from the saved per-chunk scores, without a new
Decision API request.

```bash
masked-ttt --model /path/to/model --request request.json \
  --score-json-dir /path/to/shared_scores --case-id YOUR_CASE_ID \
  --context-reserve-tokens 1024 --local-files-only --output result.json
```

The full answer is `prediction`; the independent selected-KV result is
`standalone_cartridge.prediction`. The `self_study`, `self_judge`, `training`,
and `memory` fields record the actual request path. If a document exceeds a
backbone's native window, the cached selector constructs a source-prioritized
working document and records its source mapping.

To run the recorded Qwen3-4B BABILONG V5 settings, add
`--config-json examples/qwen_babilong_v5_config.json` to the command above.
That file preserves the saved experiment's 30% selection, rollout, Teacher
judgment and optimizer settings. Explicit command-line options override the
corresponding JSON values. The same local backbone, request formatting and
cached Decision scores are needed to reproduce the experiment; exact outputs
can still depend on the execution environment.

For the GPT-OSS selected-cache development candidate, use
`--config-json examples/gpt_oss_babilong_selected_cache_candidate.json`.
The candidate keeps the practice token-boundary fix enabled. Its output policy
is explicitly configurable as `original`, `teacher_format`, or
`practice_gated_format`. The gated policy selects the trained answer only
when the optimizer actually used an independently verified practice question;
otherwise it returns the zero-update answer using the same Teacher-format
instruction. No Teacher answer or evaluation reference is inserted into the
student prompt. Development results determine which policy, if any, is used
on the held-out cases.

The original implementation used for the earlier 10% Qwen, OLMo and GPT-OSS
base experiments remains available at the Git tag `babilong-v1`. The current
code includes the later V5 self-study and optional Teacher-judgment paths, so
the tag should be used when reproducing those earlier base runs.

## Verify

Run `pytest -q`. Evaluation compares paired answers with the same task-aware
metric, reports gains and losses relative to each baseline, and reports
document KV and cartridge storage separately. Earlier OPD-only and
teacher-prefix runs are different configurations, not results of this method.
The CLI integration test builds a tiny local model and score JSON, then checks
the complete cached-selection and adaptation path without a network call.
