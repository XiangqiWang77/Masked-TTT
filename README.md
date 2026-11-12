<!-- contextual-acl-package -->
# Contextual Cartridges

**Query-Guided Test-Time Learning with Selective Self-Distillation**

[ACL paper and complete results package](paper/contextual_cartridges/README.md) · [Paper PDF](paper/contextual_cartridges/paper.pdf) · [完整中文方法](paper/contextual_cartridges/METHOD.zh-CN.md) · [BABILong 待填结果表](paper/contextual_cartridges/RESULTS.md)

1,000 BABILong requests; 26,451 count-matched synthetic exercises per method. Three separate experimental figures, using only axes, legends, and the blue–cyan / yellow–orange palette. No bars or confidence bands.

![Source dilution and all-fact co-exposure](paper/contextual_cartridges/figures/figure1_dilution_fragmentation.png)

![Task-by-length paired contrasts](paper/contextual_cartridges/figures/figure2_task_length.png)

![Paired source recall and source footprint](paper/contextual_cartridges/figures/figure3_paired_coverage_footprint.png)

[Definitions, denominators, figure PDFs/SVGs and source JSON](paper/contextual_cartridges/README.md). These completed data audits measure source coverage and organization; final answer accuracy is not inferred from them.

[Three backbones × three datasets cache–performance template](paper/contextual_cartridges/figures/figure4_cache_grid_pending.pdf). Downstream tables and cache curves are explicitly pending; no missing result is filled with zero or an older run.

[Reproduction code and software references](paper/contextual_cartridges/REPRODUCE.md).

<!-- synthesis-time-violin -->
## Synthetic data generation time

![Vanilla and query-specific synthesis-time distributions with median lines](paper/contextual_cartridges/synthesis_time/synthesis_time_violin.png)

Two violins, each using the same 635 BABILong requests with matched generated train/validation counts
(20,272 QA pairs per method). Horizontal lines mark medians: Vanilla **137.09 s**, query-specific
**126.02 s**. The query-specific tail is longer; historical hardware/concurrency and cached evidence
selection differ, so this is not a controlled end-to-end speedup claim.

[Figure PDF/SVG, raw timing CSV, definitions and reproduction code](paper/contextual_cartridges/synthesis_time/README.md).
<!-- /synthesis-time-violin -->

<!-- synthetic-three-datasets-records -->
## Synthetic-record diagnostics: LongHealth, MTOB (KE), QASPER

![Matched query-term focus and non-adjacent source exposure for the synthetic datasets used by Vanilla and Contextual Cartridges](paper/contextual_cartridges/synthetic_three_datasets/synthetic_dilution_fragmentation.png)

**107 fixed requests; 642 matched exercise exposures per method.** These two panels describe
lexical query focus and source organization in the saved records, not model accuracy or verified
multi-hop reasoning. Vanilla's zero in the second panel follows from this run's contiguous-window
construction. Existing query fallbacks are included; a paired analysis excluding their 22 requests
is also reported. This cohort is separate from the historical BABILong timing comparison above.

[Definitions, tables, fallback sensitivity and limitations](paper/contextual_cartridges/synthetic_three_datasets/README.md) ·
[PDF](paper/contextual_cartridges/synthetic_three_datasets/synthetic_dilution_fragmentation.pdf) ·
[Record JSONL (gzip)](paper/contextual_cartridges/synthetic_three_datasets/records.jsonl.gz) ·
[Reproduction script](paper/contextual_cartridges/synthetic_three_datasets/reproduce.py).
<!-- /synthetic-three-datasets-records -->

<!-- previous-report -->
[Historical report (previous protocols)](archive/README_before_acl_20261111.md)

