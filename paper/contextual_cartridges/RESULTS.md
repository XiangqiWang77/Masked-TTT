# BABILong result tables — to be filled

The paper reserves 1,000 paired requests per backbone (ten tasks, four lengths). All accuracy cells below are pending. A dash is unavailable, not zero. V-SD denotes ordinary self-distillation and S-SD the current conservative TracIn-guided selector. The selective delta compares S-SD against V-SD on the same query-specific dataset. No old QC or earlier TracIn accuracy is reused.

**Overall accuracy (%)**

| Model | Full-context ICL | Agnostic + V-SD | Agnostic + S-SD | Specific + V-SD | Contextual (Specific + S-SD) | Δ selective (pp) |
|---|---|---|---|---|---|---|
| Qwen3-4B | — | — | — | — | — | — |
| OLMo-3-7B | — | — | — | — | — | — |
| GPT-OSS-20B | — | — | — | — | — | — |


**Accuracy by document length (%) — 250 requests per cell**

| Model | Length | Full-context ICL | Agnostic + V-SD | Agnostic + S-SD | Specific + V-SD | Contextual (Specific + S-SD) | Δ selective (pp) |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 8K | — | — | — | — | — | — |
| Qwen3-4B | 16K | — | — | — | — | — | — |
| Qwen3-4B | 32K | — | — | — | — | — | — |
| Qwen3-4B | 64K | — | — | — | — | — | — |
| OLMo-3-7B | 8K | — | — | — | — | — | — |
| OLMo-3-7B | 16K | — | — | — | — | — | — |
| OLMo-3-7B | 32K | — | — | — | — | — | — |
| OLMo-3-7B | 64K | — | — | — | — | — | — |
| GPT-OSS-20B | 8K | — | — | — | — | — | — |
| GPT-OSS-20B | 16K | — | — | — | — | — | — |
| GPT-OSS-20B | 32K | — | — | — | — | — | — |
| GPT-OSS-20B | 64K | — | — | — | — | — | — |


[Fill the source JSON](data/babilong_results.json) and rebuild with `python code/build_tables.py`. The renderer requires completed denominators and a result source before filling accuracy.

[Overall LaTeX](tables/babilong_overall.tex) · [Length LaTeX](tables/babilong_by_length.tex) · [Paper](paper.pdf)

**Cache curves.** The [36-row cache template](data/cache_results.json) reserves three backbones × three datasets × four sizes. LongHealth uses accuracy; MTOB (KE) uses chrF; QASPER uses log-perplexity. All values remain null in this snapshot. [Nine-panel layout](figures/figure4_cache_grid_pending.pdf).
