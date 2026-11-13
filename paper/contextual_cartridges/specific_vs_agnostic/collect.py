"""Why query-specific data beats query-agnostic data (Qwen3-4B, synthetic_threeway_min15 + fast_v2 SD).

Per practice question: query-term recall (same word/stopword rules as the dilution audit).
Per training answer: share of answer words that are query content terms.
Per request: teacher contexts with gold evidence (reranker audit) and paired SD outcome.
"""
import csv, glob, json, re, sys
from pathlib import Path
sys.path.insert(0, '/home/da839/Query_Cart')
from experiments.run_figure4_contextual import load_cases, longhealth_score

DATA = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_cartridges_figure4_datasets_20261110'
SYN = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/synthetic_threeway_min15_20261112/synthetic')
FAST = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/results')
SPEC = json.loads(Path('/home/da839/Query_Cart/runs/github_dilution_coverage_20261112/metric_specification.json').read_text())
STOP = set(SPEC['stopwords'])
words = lambda s: [w for w in re.findall(r'\w+', s.casefold()) if w not in STOP and len(w) > 1]
plan = json.loads(Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/plan.json').read_text())
requests = sorted({(r['dataset'], r['case']) for r in plan})
cases = {d: {c['id']: c for c in load_cases(d, DATA)} for d in ('longhealth', 'mtob_ke', 'qasper')}
qrows, outcome = [], []
for d, c in requests:
    case = cases[d][c]
    qterms = set(words(case['query']))
    if not qterms:
        continue
    for arm in ('specific', 'agnostic'):
        f = SYN / d / c / f'{arm}.json'
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        items = data['examples'] if arm == 'specific' else data['questions']
        for item in items[:15]:
            qw = set(words(item['question']))
            ans = words(item.get('teacher_answer') or item.get('answer') or '')
            qrows.append(dict(dataset=d, case=c, arm=arm, recall=len(qterms & qw) / len(qterms),
                              answer_query_density=(sum(w in qterms for w in ans) / len(ans)) if ans else 0.0,
                              answer_words=len(ans)))
    for b in ('r05', 'r10', 'r15', 'r20'):
        fs, fa = FAST / 'specific_sd' / d / b / f'{c}.json', FAST / 'vanilla_cartridges' / d / b / f'{c}.json'
        if fs.exists() and fa.exists():
            rs, ra = json.loads(fs.read_text()), json.loads(fa.read_text())
            val = lambda r: float(longhealth_score(r.get('prediction') or '', case)) if d == 'longhealth' else float(r['metric_value'])
            outcome.append(dict(dataset=d, case=c, budget=b, specific=val(rs), agnostic=val(ra)))
out = Path(__file__).parent
for name, rows in (('questions.csv', qrows), ('outcome.csv', outcome)):
    with (out / name).open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(qrows), 'questions;', len(outcome), 'paired cells')
