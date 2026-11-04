# Comparing one Value patch in the two caches

For the same selected source token, the full-document prefill produces a Value vector \(v_D\); the independently prefetched selected text produces \(v_E\). These may differ because their preceding contexts differ. The original OLMo-style method learns a per-chunk Value correction on the full view and applies the same basis and coefficients to the selected view. The relevant question is therefore not just whether the selected vectors have energy in the basis, but whether the **realized correction** is similar on both inputs.

[`value_alignment_probe.py`](value_alignment_probe.py) maps selected tokens from the full prompt to the independently prefetched evidence cache, checks that their token identities agree, and excludes attention sinks. It samples at most 256 positions deterministically across the selected document. For each adapted layer and KV head it reports:

* The cosine between \(v_D\) and \(v_E\), and between their low-rank coordinates.
* The cosine between the written corrections, plus their sizes relative to the original Values and their nonzero-component fractions.
* The selected Value energy within the original basis and the number of valid comparisons.

The written corrections use the production `KVReadMemory._chunk_values` implementation on sampled tokens, including per-chunk coefficients, the Value-radius bound, cache-dtype conversion and addition. Sampling may change kernel batch shapes, so these numbers are not guaranteed bitwise identical to a complete-cache write. Zero-length vectors and updates have undefined cosine, reported as null rather than zero.

The observer runs after SD while both original caches and the trained state still exist. It uses no labels, new loss, optimizer update or API; it releases all captured objects before independent cartridge decoding. It patches request-local constructors and is intended for a single-request process. A small-model check covers Qwen3 and GPT-OSS together with the gain and first-answer observers, and verifies unchanged answer/gradient paths and physical cache release. A real-model GPU check is still needed before interpreting any measured cosine.

```python
from experiments.value_lr_physical.value_alignment_probe import capture_value_alignment
from masked_ttt.kv_only import kv_only_cartridge

with capture_value_alignment(max_positions=256) as observed, kv_only_cartridge():
    result = method.answer(request)
statistics = observed["statistics"]
```

The diagnostic driver accepts an optional `--value-alignment` flag and includes a loaded-model observation-neutrality preflight. The preparation script accepts the same flag, freezes it into the manifest and Slurm invocation, and uses a separate `query_cartridges_gain_value_transfer_20261104` directory. Preparation does not submit without `--submit`. Already submitted frozen jobs retain their original code and do not automatically acquire this probe.

[Earlier 40-case basis ablation](../../docs/analysis/value-basis-ablation-20261104/README.md) · [Tests](../../tests/test_value_alignment_probe.py)
