# Synthetic data generation time

The left pair of violins compares historical Qwen3-4B BABILong runs. The right pair (added 2026-11-13) is
OLMo-3-7B-Instruct on LongHealth from the current full-table campaign.
Each observation is the measured time to construct the synthetic dataset for one request. It is not TTFT.

The figure contains 635 matched requests with identical generated train and validation
question counts (20,272 QA pairs per method). All 1,000 records and
their exclusion reasons are preserved in `synthesis_time_per_request.csv`. Eligibility never uses timing
advantages, benchmark answers, or model accuracy. Sixteen query-specific records include checkpoint
replay and are excluded from the primary comparison. Other exclusions reflect unequal generated counts.

| Method | Mean (s) | Median (s) | P90 (s) |
|---|---:|---:|---:|
| Vanilla synthesis | 129.47 | 137.09 | 155.37 |
| Query-specific synthesis | 136.08 | 126.02 | 275.69 |

Query-specific synthesis is faster on 348/635 matched requests.
Its median is lower, but its mean and P90 are higher. These observations do not establish uniform speed superiority.

## OLMo-3-7B-Instruct (LongHealth)

| Method | Mean (s) | Median (s) | P90 (s) |
|---|---:|---:|---:|
| Vanilla synthesis | 75.13 | 70.48 | 113.26 |
| Query-specific synthesis | 133.79 | 116.92 | 200.02 |

124 requests, each with exactly 15 vanilla and 15 query-specific questions. Query-specific synthesis
is faster on 17/124 requests. On OLMo the query-specific pipeline
(pseudo-answer, then question, then Teacher answer) is slower than vanilla self-study across the
distribution, unlike the Qwen BABILong runs, where the query-specific median was lower.

How the OLMo times were measured: both arms of a request run back-to-back in one coroutine on one
H200 vLLM engine shared with other in-flight requests. No per-arm timer was logged. Per-arm wall time
is recovered from the segment start (`pair.json` mtime minus `resumed_segment_seconds`) and the
mtimes of the two target files; see `olmo_synthesis_time.py`. A request is eligible only when both
arms were generated from scratch inside job 33294262, the first job whose answer prompt OLMo could
follow. All QASPER and MTOB requests, and the remaining LongHealth requests, had earlier partial
progress or crossed a job boundary, so they are excluded. All 283 records and their exclusion
reasons are in `olmo3_synthesis_time_per_request.csv`. Eligibility never uses timing values,
benchmark answers or accuracy.

The two backbones differ in model, dataset, questions per arm, concurrency and hardware. Compare
vanilla with query-specific within a backbone, not across backbones.

![Four violins: Qwen3-4B BABILong (left) and OLMo-3-7B LongHealth (right). Each dark horizontal line is the sample median.](synthesis_time_violin.png)

[PDF](synthesis_time_violin.pdf) · [SVG](synthesis_time_violin.svg) · [Qwen timing CSV](synthesis_time_per_request.csv) · [OLMo timing CSV](olmo3_synthesis_time_per_request.csv) · [Summary JSON](synthesis_time_summary.json)

Timing includes shared-backend waiting. Concurrency differed (Vanilla 32 versus query-specific 16 cases
per worker), hardware was not held fixed, and the query-specific run reused prior Decision API evidence
selection. Those historical API calls are not included. These are observational run-time distributions,
not a controlled same-hardware speed benchmark. Do not sum per-request durations to infer campaign wall
time: concurrent requests overlap. No runtime changes or GPU submissions were made for this analysis.

Each violin uses all eligible durations, with Scott KDE bounded by the observed minimum and maximum.
Width encodes density, not sample count. The dark horizontal line is the sample median. The original
long tail remains visible. Individual dots, bars, and confidence intervals are not displayed. CSV, JSON,
PDF, PNG, SVG and the reproducible analysis script accompany the figure. General research style;
no ACL compliance certification.

Software: the scientific-visualization skill informed figure integrity and provenance checks.
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026),
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).

Reproduce the figure from the published CSVs with `python replot.py` (Qwen check) then `python plot_qwen_olmo_violin.py`; `olmo_synthesis_time.py` rebuilds the OLMo CSV from the campaign directory (Python 3.11+, NumPy, Matplotlib; versions recorded in the summary JSON). The full source-record analysis is in `plot_synthesis_time_distribution.py`.
