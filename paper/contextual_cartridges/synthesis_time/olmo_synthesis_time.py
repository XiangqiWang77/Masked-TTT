"""Collect per-request OLMo-3-7B-Instruct synthesis times from the full-table campaign.

The OLMo pipeline builds both arms for one request inside one coroutine:
query-specific first (pseudo-answer -> question -> Teacher answer), then
query-agnostic (Vanilla Cartridges self-study). No per-arm timer was logged,
so per-arm wall time is recovered from the run's own records:

  segment start      = mtime(pair.json) - pair.json['resumed_segment_seconds']
  specific seconds   = mtime(specific.targets.pkl) - segment start
  vanilla seconds    = mtime(agnostic.targets.pkl) - mtime(specific.targets.pkl)

This is valid only when both arms were generated from scratch inside that one
segment. Records are therefore eligible only if (a) the pair is complete with
15+15 questions, (b) the segment ran in job 33294262, the first job whose
answer prompt OLMo could follow, (c) the earlier job 33288329 finished the
request with 0 accepted specific questions and no agnostic data, and (d) the
agnostic targets were written after the specific targets. Everything else is
kept in the CSV with its exclusion reason. Eligibility never looks at timing
values, benchmark answers or accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

JOBS = {  # sacct Start/End, America/New_York
    '33288329': ('2026-11-13T09:54:43-05:00', '2026-11-13T10:46:05-05:00'),
    '33294262': ('2026-11-13T10:46:58-05:00', '2026-11-13T13:20:23-05:00'),
    '33309195': ('2026-11-13T13:20:40-05:00', '2026-11-13T19:20:40-05:00'),
}
TIMED_JOB = '33294262'
FIELDS = ['dataset', 'case', 'in_175_request_table', 'segment_job', 'vanilla_seconds', 'specific_seconds',
          'vanilla_questions', 'specific_questions', 'primary_eligible', 'exclusion_reasons']


def window(job):
    return tuple(datetime.fromisoformat(t).timestamp() for t in JOBS[job])


def first_job_outcomes(campaign):
    out = {}
    for log in sorted((campaign / 'logs').glob('worker-*-33288329.out')):
        for line in log.read_text().splitlines():
            if '"synthesis_request_finished"' not in line:
                continue
            d = json.loads(line)
            m = re.search(r'Specific synthesis shortfall: (\d+)/', ' '.join(d['errors']))
            out[(d['dataset'], d['case'])] = ('complete' if not d['errors'] else
                                              f'specific_{m.group(1)}' if m else 'other_error')
    return out


def collect(campaign):
    table = {(r['dataset'], r['case']) for r in json.loads((campaign / 'requests.json').read_text())}
    first = first_job_outcomes(campaign)
    rows = []
    for pair_path in sorted((campaign / 'synthetic').glob('*/*/pair.json')):
        root = pair_path.parent
        dataset, case = root.parent.name, root.name
        pair = json.loads(pair_path.read_text())
        counts = json.loads(pair['actual_counts'].replace("'", '"')) if isinstance(pair['actual_counts'], str) else pair['actual_counts']
        start = pair_path.stat().st_mtime - float(pair['resumed_segment_seconds'])
        job = next((j for j in JOBS if window(j)[0] <= start <= window(j)[1]), '')
        t_spec = (root / 'specific.targets.pkl').stat().st_mtime
        t_agn = (root / 'agnostic.targets.pkl').stat().st_mtime
        reasons = []
        if str(pair['complete']) != 'True' or counts != {'specific': 15, 'agnostic': 15}:
            reasons.append('incomplete')
        if job != TIMED_JOB:
            reasons.append('segment_not_in_timed_job')
        prior = first.get((dataset, case), 'no_finished_record_in_first_job')
        if prior != 'specific_0':
            reasons.append(f'prior_progress_or_unknown:{prior}')
        if not start <= t_spec <= t_agn:
            reasons.append('arms_not_both_written_in_segment')
        rows.append(dict(dataset=dataset, case=case, in_175_request_table=(dataset, case) in table,
                         segment_job=job, vanilla_seconds=t_agn - t_spec, specific_seconds=t_spec - start,
                         vanilla_questions=counts.get('agnostic'), specific_questions=counts.get('specific'),
                         primary_eligible=not reasons, exclusion_reasons=';'.join(reasons)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', type=Path, default=Path(
        '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/olmo3_full_table_20261113'))
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'olmo3_synthesis_time_per_request.csv')
    args = ap.parse_args()
    rows = collect(args.campaign)
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    print(f'{sum(r["primary_eligible"] for r in rows)}/{len(rows)} eligible -> {args.output}')


if __name__ == '__main__':
    main()
