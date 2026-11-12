"""Reproduce the two violins from the published CSV; no GPU or private data."""
import csv
import json
from pathlib import Path

from plot_synthesis_time_distribution import describe, figures

root = Path(__file__).resolve().parent
with (root / 'synthesis_time_per_request.csv').open() as stream:
    records = list(csv.DictReader(stream))
rows = []
for record in records:
    if record['primary_eligible'] != 'True':
        continue
    row = dict(record)
    for key in ('vanilla_seconds', 'specific_seconds'):
        row[key] = float(row[key])
    for key in ('vanilla_questions', 'specific_questions', 'document_tokens'):
        row[key] = int(row[key])
    rows.append(row)
summary = json.loads((root / 'synthesis_time_summary.json').read_text())
observed = describe(rows)
assert observed['n'] == summary['primary']['n']
for method in ('vanilla', 'query_specific'):
    for metric in ('mean_seconds', 'p50_seconds', 'p90_seconds'):
        assert abs(observed[method][metric] - summary['primary'][method][metric]) < 1e-9
figures(rows, root)
print(f'Reproduced two violins from {len(rows)} paired requests.')
