# Selected-cache transplant check

The [materialization tests](../../../tests/test_materialization_equivalence.py)
construct a full-document KV and a separately prefetched selected-text KV,
with two nonadjacent selected chunks and distinct learned read parameters.
They calculate expected materialized keys and values directly from the
original trained parameters and the selected cache, then compare every
position. This checks that each chunk receives its own gain and value map,
and that prefixes and separators remain unchanged. The original full and
selected caches must remain unchanged.

The Qwen and grouped-query GPT-OSS CPU cases passed; CUDA variants are ready
for a GPU environment. This validates the transplant calculation on small
models and does not itself measure BABILONG performance.
