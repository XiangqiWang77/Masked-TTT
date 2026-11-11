# Data, code, and reproduction

The authoritative figure input is [synthetic_source.json](data/synthetic_source.json), copied byte-for-byte from the saved 1,000-request, 26,451-exercise-per-arm audit snapshot. It contains per-request metrics and source hashes, not hidden benchmark answers. [Per-request CSV](data/synthetic_per_request.csv), [aggregate JSON](data/synthetic_summary.json), [heatmap cells](data/heatmap_cells.csv), and [figure manifest](data/figure_manifest.json) make every plotted value reproducible. All requests, including regressions, are retained. The only metric-specific exclusion is the six unavailable paired recall values.

Full synthetic examples remain in the existing [Answer-Guided v3 release](https://github.com/XiangqiWang77/Masked-TTT/releases/tag/contextual-cartridges-answer-guided-v3-20261109); the [release index](../../datasets/babilong_synthetic_qwen3_4b/answer_guided_v3_release.json) identifies the two source datasets. Do not substitute the older 27,949-exercise audit for this count-matched 26,451-exercise snapshot. "1,000 requests" counts document–query pairs across four lengths, not 1,000 independent source episodes.

From this directory, using Python with Matplotlib, NumPy, and Pillow:

```bash
python code/build_figures.py
python code/build_tables.py
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

On the current Bouchet environment, the build used `matplotlib/3.9.2-gfbf-2024a` and `texlive/20240312-GCC-13.3.0`. The figures are 160.02 mm wide, exported as vector PDF, editable SVG, and 400-DPI PNG. The table code uses ordinary Python without model or GPU dependencies. No training or scheduler command is invoked by either build script.

Fill [babilong_results.json](data/babilong_results.json) and [cache_results.json](data/cache_results.json) only from frozen-protocol result files. Unknown values stay `null`, and incomplete rows stay `pending`. BABILong rows become complete only with the full denominator and a traceable source. Cache points require a common `paired_case_set_sha256` across capacities; missing points stay gaps. Then rebuild tables and figures. If results have been filled, revise the paper's pending captions to match; the publication does not auto-promote a partial run to a final result.

The [implementation snapshot](code/implementation/README.md) documents the current selector, objective, and runner, with hashes. It is a review snapshot of existing project modules, not a standalone reimplementation of the upstream library. The draft deliberately distinguishes the current confirmed-deletion selector from the historical summed-score selector.

The task metrics follow [Cartridges](https://arxiv.org/abs/2506.06266). QASPER's log-perplexity and MTOB's chrF are not accuracy percentages. Published baseline scores must carry their original backbone, evaluation split, metric, and source; an LLaMA result must not be relabeled as a Qwen or OLMo result.

The plotting and scientific-integrity workflow used **scientific-visualization**, described by Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). The current arXiv record was checked on 2026-11-11. The workflow does not certify scientific validity or conference acceptance.
