# Diagnosing transfer of the attention update

The original method learns selected-chunk gains while the Student can also attend to unselected document positions. The deployed cartridge instead reads an independently prefetched selected document. A uniform increase across all selected chunks can move attention from unselected to selected document positions during training; among the selected document positions alone, the same increase cancels under softmax. In the deployed model, suffix tokens, separators and attention sinks remain in the denominator, so a common increase can still affect their relative attention. The Value update is a separate mechanism.

[`gain_transfer_probe.py`](gain_transfer_probe.py) records, separately for each optimization step, the selected gains' gradient, actual post-Adam-and-clamp update and final parameter value. For each layer and head, it decomposes each quantity into:

* **Common:** the same value across every active selected chunk.
* **Contrast:** the remaining differences among active selected chunks.

The report gives both norms and the fraction of squared norm in the common component. This decomposition describes the update; it is not itself a measure of correctness or a claim that a component failed to transfer. Compare it with a same-prefix full-versus-cartridge readout and paired answer outcomes before choosing a change. Only chunks with at least one addressable, nonsink token are counted.

The observer is optional and intended for an isolated single-request process:

```python
from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer
from masked_ttt.kv_only import kv_only_cartridge

with capture_gain_transfer() as steps, kv_only_cartridge():
    result = method.answer(request)
```

It modifies neither the objective nor the gradient, learns no additional parameters, uses no labels or API, and stores no KV or training tensor after the update. The CPU tests verify the softmax distinction and compare instrumented versus ordinary training and standalone decoding on small Qwen3 and GPT-OSS models. A loaded-model GPU preflight and real cases are still required; frozen jobs already running do not include this new observer. The existing 40-case first-answer readout analysis is a separate experiment and motivates, rather than validates, this decomposition.

The companion [driver](run_gain_transfer.py) records the same measures on the fixed 40-case development probe for both backbones. It uses the original score-ranked 10% Decision selection, eight-token Student rollouts, two base SD updates plus verified practices, and the original document Value basis. It records actual post-clamp Adam steps and four first-answer distributions, then releases the full and selected source caches before generating the independent KV-only answer. A boundary-safe verifier fallback and the physical cache-release path are implementation fixes also used in the associated development run. Each loaded backbone first checks that observation preserves the Teacher, Student trajectory, gradients and answer. If a first-token readout cannot fit, it is marked unavailable rather than used as a zero.

For a future frozen run, the driver's optional `--value-alignment` flag adds the [paired Value diagnostic](VALUE_ALIGNMENT_PROBE.md) with the same observation-neutrality preflight. Existing submitted jobs keep their original frozen code and are unaffected.

The [preparation script](prepare_gain_transfer.py) freezes the code and protocol, runs CPU checks, and submits one low-priority GPU job under the shared 12-GPU limit. It reuses cached Decision JSON, has a six-hour cumulative GPU budget and writes partial aggregate reports as cases finish. This is a diagnostic on previously selected development cases, not a new 1,000-case accuracy estimate.

[Tests](../../tests/test_gain_transfer_probe.py) · [Existing Qwen/GPT portability diagnosis](../../docs/analysis/gain-transfer-readout-20261104/README.md)
