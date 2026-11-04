# Query Cartridges: Learning to Read Long Contexts with Masked Test-Time Training

Query Cartridges, also referred to here as Masked TTT Cartridges, adapts a
frozen language model to one long-context question.
Question-conditioned Decision scores select a small number of document chunks.
A frozen, selected-context ICL teacher supplies on-policy next-token
distributions and proposes short source-grounded practice questions. The
full-context student jointly adapts two request-local reading corrections:
chunk-level attention allocation and a low-rank value-read adjustment. Model
weights and the original document KV remain immutable.

This branch preserves the training and materialization used in the OLMo BABILONG
run. The public result interface returns the standalone answer by default; the
full-context answer is retained as a diagnostic. The recorded profile is at
[examples/olmo_babilong_icl_sd.json](examples/olmo_babilong_icl_sd.json).
The [Chinese method description](docs/OLMO_ICL_SELF_DISTILLATION.zh-CN.md)
explains the two cache paths, the paired results, and their scope.
[Three paired BABILONG examples](docs/whole_selective_case_examples.zh-CN.md)
illustrate their complementary behavior and remaining errors.

The trained state supports two explicitly different answer paths:

* **Full KV + Query Cartridge:** answer from the working-document cache with
  the trained corrections. A frozen full-context judge compares the native and
  adapted answers in both orders; an inconclusive comparison retains the native
  answer.
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

## Procedure

1. Load the saved Decision API scores for the document and question. Select
   the top 10% of source chunks in their original order.
2. Prefill the complete working document once. Separately prefill the selected
   text for the ICL teacher; obtain its answer for the baseline comparison.
3. Ask the teacher for up to two short factual practice questions, answers and
   exact source quotations. Keep only proposals whose quotation occurs in a
   selected chunk and whose answer agrees with a separate frozen-model answer
   based on that quotation alone. This is a consistency check, not a guarantee
   of logical entailment; evaluation references are never used here.
4. Generate short current-student prefixes for the original and verified
   practice questions. Score each identical prefix with the focused teacher and
   minimize full-vocabulary forward KL on the student path. All exercises share
   the same selected-position attention and value-read state.
5. Discard training suffixes and, by default, release the full-document KV
   before generating the original answer from the independent selected-KV
   cartridge. A separately recorded full-context diagnostic can use a frozen
   judge to compare native and adapted answers. That judge does not select or
   alter the standalone cartridge's answer.

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

For standalone decoding, the implementation releases the training adapter and,
by default, the selected source cache after materialization. The cartridge
keeps any storage it still needs. `standalone_cartridge_bytes` is the deployable
artifact size; `live_document_state_before_decode_bytes` additionally counts
any training or source tensors still live, with shared storage counted once.
Both exclude model weights and question/output suffix KV. The
`additional_source_kv_bytes_before_decode` and
`additional_training_adapter_bytes_before_decode` fields report any retained
storage outside the artifact. This runtime count is distinct from construction
peak.

## Run

Install with `pip install -e '.[test]'` and a compatible FlashAttention build.
The saved score directory contains `case_to_score_file.json` and `scores/`.

```bash
masked-ttt --model /path/to/model --request request.json \
  --score-json-dir /path/to/shared_scores --case-id YOUR_CASE_ID \
  --config-json examples/olmo_babilong_icl_sd.json \
  --local-files-only --output result.json
```

`prediction`, `generated_token_ids`, and `finish_reason` describe the primary
output. With standalone enabled, they match `standalone_cartridge` and
`output_source` is `standalone_cartridge`; the paired full-context answer is
available in `full_context`. With standalone disabled, explicitly set both
`standalone_cartridge_enabled` and `release_full_before_cartridge` to `false`;
the primary answer then comes from the full-context path. `self_judge.applies_to`
identifies which path the judge can affect.

`timing.ttft_seconds` measures through the first token of the primary output,
including preceding online work and selected-cache materialization. Model
loading and any previously cached offline Decision scoring are excluded;
`full_context_ttft_seconds` records the separate full-context diagnostic.
The `self_study`, `self_judge`, `training`, and `memory` fields record the
actual request path. If a document exceeds a backbone's native window, the
cached selector constructs a source-prioritized working document and records
its source mapping.

### Optional linked selection and KV-only decoding

The same request path supports Qwen3-4B and GPT-OSS-20B with cached Decision
scores. The experimental linked selector keeps the same chunk budget while
reserving some slots for related source context. The KV-only option discards
the duplicate read-state tensors after they have been materialized into the
selected cache.

```python
from masked_ttt import Backbone, MaskedTTT, Request, TTTConfig
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.linked_selection import LinkedFrozenSelector

config = TTTConfig(
    keep_fraction=0.10,
    max_context=131072,
    practice_boundary_fallback=True,
    selected_transfer_weight=0.0,
    release_full_before_cartridge=True,
    release_selected_source_before_decode=True,
)
# Also accepts a local GPT-OSS-20B checkpoint.
backbone = Backbone.load("/path/to/Qwen3-4B-Instruct-2507", config, local_files_only=True)
selector = LinkedFrozenSelector("/path/to/shared_scores")
selector.bind_case("your-case-id")
request = Request(document=document, question=question)

with kv_only_cartridge():
    result = MaskedTTT(backbone, selector).answer(request)

print(result["prediction"])
print(result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"])
```

The document, question and case ID must match the saved scores. No online API
call is made by this selector. `kv_only_cartridge()` temporarily patches
process-local decoding hooks, so use one request at a time in that process.
For a benchmark reproduction, pass its original task prompt and generation
limit to `Request`; the example's default prompt is only a general-purpose
starting point.
GPT-OSS expert weights are dequantized to preserve input gradients; the GPU
memory required depends on context length. The linked policy is still being
validated beyond the development set.

## Verify

Run `pytest -q`. Evaluation compares paired answers with the same task-aware
metric, reports gains and losses relative to each baseline, and reports
document KV and cartridge storage separately. Earlier OPD-only and
teacher-prefix runs are different configurations, not results of this method.
