# Native-KV TracIn-Selective SD protocol snapshot

This directory records the frozen four-arm development protocol for **Contextual Cartridges: Query-Guided Test-Time Learning with Selective Self-Distillation**. The synthetic datasets are complete. Downstream claims that Query-Specific data or Selective SD improve BABILong accuracy remain pending until the complete paired run finishes.

[Full method](../../../CONTEXTUAL_CARTRIDGES_METHOD.zh-CN.md) · [Evaluation protocol](../../../CONTEXTUAL_CARTRIDGES_EVALUATION.md) · [Dataset release manifest](../answer_guided_v3_release.json)

The frozen matrix is:

1. `vanilla_syns_vanilla_sd`
2. `vanilla_syns_selective_sd`
3. `specific_syns_vanilla_sd`
4. `specific_syns_selective_sd`

Selective SD uses native-Cartridge gradients only. It computes leave-one-out signed TracIn at the initial and trained Vanilla-SD checkpoints, retains candidates whose summed score is strictly greater than zero, and uses those candidates only for second-half consolidation. It uses no held-out labels or benchmark answers. If every candidate is rejected, the exact trained Vanilla checkpoint is returned.

`configuration.cluster.json` is the exact frozen cluster configuration, including the 160 development case list and cluster-local paths. `configuration.json` is the concise portable protocol summary. The `code/` directory is a review snapshot of changed modules and tests, not a standalone installation. It depends on the Query_Cart workspace and Stanford Hazy Research's official `cartridges` checkout at `ef34ba97a06049c34820506e2c283746284ae5f0`.

Run the dependency-light protocol tests from the repository root with the snapshot on `PYTHONPATH`:

```bash
CC_CODE=datasets/babilong_synthetic_qwen3_4b/sd_protocol/code
PYTHONPATH="$CC_CODE" pytest -q "$CC_CODE/tests"
```

The plotting test additionally requires Matplotlib; the TracIn gradient test requires PyTorch. In the publication audit these were executed in their corresponding cluster modules, for a total of 47 passing tests.
