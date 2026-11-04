# Hourly Query Cartridge checks

`due_check.py` reads the existing per-job schedule before querying Slurm or
reading fresh reports. A job is inspected only at or after its recorded
`next_check_not_before_utc`; a query failure stays `UNKNOWN` and is retried on
the next interval. Missing queue output alone is not treated as termination:
the existing Slurm accounting/requeue guard must confirm it.

When a development job is due, the check can register an already submitted
source-disjoint holdout without querying the new job early. Development jobs
with automatic promotion remain scheduled until the hourly monitor has
resolved promotion, even if the development allocation has completed.
Summaries contain aggregate metrics and report timestamps, not raw questions or
answers. The helper starts no jobs, calls no API and never changes training.

```bash
/usr/bin/python3 -B experiments/monitoring/due_check.py --workspace /path/to/cross_backbone_selective_sd_20261102
```

[Tests](../../tests/test_due_check.py) exercise interval enforcement, unknown
states, confirmed termination and holdout registration with synthetic data.
