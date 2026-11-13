"""stats.json -> macros.tex (numbers used in prose) and table_main.tex."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'stats.json').read_text())
EVID = json.loads(Path('/home/da839/Query_Cart/runs/reranker_evidence_accuracy_20261112/summary.json').read_text())
COVER = json.loads(Path('/home/da839/Query_Cart/runs/github_dilution_coverage_20261112/summary.json').read_text())
DS = {'longhealth': 'LH', 'mtob_ke': 'MT', 'qasper': 'QA'}
ARMS = {'specific_vs_agnostic_SD': 'SpecVsAgn', 'dualjsd_vs_SD_agnostic': 'DualAgn',
        'dualjsd_vs_SD_specific': 'DualSpec', 'specific_dualjsd_vs_vanilla': 'DualSpecVsVan'}
lines = []
def macro(name, value):
    lines.append(f'\\newcommand{{\\{name}}}{{{value}}}')
for d, tag in DS.items():
    for key, arm in ARMS.items():
        p = S['paired'][d][key]
        macro(f'{arm}{tag}W', p['wins']); macro(f'{arm}{tag}L', p['losses']); macro(f'{arm}{tag}T', p['ties'])
        macro(f'{arm}{tag}N', p['requests']); macro(f'{arm}{tag}C', p['cells'])
        g = p['mean_gain']; macro(f'{arm}{tag}G', 'n/a' if g != g else f'{g:+.2f}')
    macro(f'Cover{tag}Spec', f"{COVER[d]['arms']['specific']['query_term_recall']:.0f}")
    macro(f'Cover{tag}Agn', f"{COVER[d]['arms']['vanilla']['query_term_recall']:.0f}")
    for arm in ('specific', 'agnostic'):
        a = S['audit'].get(f'{d}/{arm}')
        if a:
            macro(f'Jsd{tag}{arm[:4].capitalize()}', f"{a['frac']:.1f}")
            macro(f'Agree{tag}{arm[:4].capitalize()}', f"{a['agree']:.0f}")
            macro(f'Weight{tag}{arm[:4].capitalize()}', f"{a['weight']:.2f}")
            macro(f'Traj{tag}{arm[:4].capitalize()}', a['trajectories'])
for d, tag in (('longhealth', 'LH'), ('qasper', 'QA')):
    e = EVID[d]
    macro(f'EvHitOne{tag}', f"{e['hit_at_k']['reranker']['1']*100:.0f}")
    macro(f'EvHitFive{tag}', f"{e['hit_at_k']['reranker']['5']*100:.0f}")
    macro(f'EvLexOne{tag}', f"{e['hit_at_k']['lexical']['1']*100:.0f}")
    macro(f'EvRandOne{tag}', f"{e['hit_at_k']['random']['1']*100:.0f}")
    macro(f'EvSpec{tag}', f"{e['fraction_specific_trajectories_with_gold']*100:.0f}")
    macro(f'EvAgn{tag}', f"{e['fraction_agnostic_trajectories_with_gold']*100:.0f}")
    macro(f'EvN{tag}', e['requests'])
a = S['audit']['mtob_ke/specific']; macro('AudMTW', f"{a['lpw']:.2f}"); macro('AudMTE', f"{a['lpe']:.2f}")
(HERE / 'macros.tex').write_text('\n'.join(lines) + '\n')

COLS = [('Whole KV', 'Whole KV'), ('Evidence Teacher', 'Evid.\\ ICL'), ('SnapKV', 'SnapKV'),
        ('FastKV (attention matching)', 'FastKV (AM)'),
        ('Vanilla Cartridges (agnostic SD)', 'Vanilla Cart.'), ('Agnostic + DualJSD (running)', 'Vanilla+DualJSD'),
        ('Specific Cartridges (specific SD)', 'Spec.+SD'), ('Specific + DualJSD (running)', 'Spec.+DualJSD (ours)')]
NAME = {'longhealth': 'LongHealth Acc.\\,$\\uparrow$', 'mtob_ke': 'MTOB chrF\\,$\\uparrow$', 'qasper': 'QASPER log-PPL\\,$\\downarrow$'}
SIGN = {'longhealth': 1, 'mtob_ke': 1, 'qasper': -1}
rows = []
for d in DS:
    for b, pct in zip(('r05', 'r10', 'r15', 'r20'), (5, 10, 15, 20)):
        e = S['main'][d][b]
        vals = {m: e.get(m, float('nan')) for m, _ in COLS}
        learned = [m for m, _ in COLS[2:] if not math.isnan(vals[m]) and not (d == 'mtob_ke' and m in ('SnapKV', 'FastKV (attention matching)'))]
        best = max(learned, key=lambda m: SIGN[d] * vals[m]) if learned else None
        cells = []
        for m, _ in COLS:
            v = vals[m]
            if math.isnan(v): cells.append('--'); continue
            text = f'{v:.1f}' if d != 'qasper' else f'{v:.2f}'
            if d == 'mtob_ke' and m in ('Whole KV', 'SnapKV', 'FastKV (attention matching)'): text += '$^\\dagger$'
            if m == best: text = f'\\textbf{{{text}}}'
            cells.append(text)
        head = f'\\multirow{{4}}{{*}}{{\\shortstack[l]{{{NAME[d]}}}}}' if pct == 5 else ''
        rows.append(f"{head} & {pct}\\% & {e['n']} & " + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule')
rows = rows[:-1]
header = ' & '.join(['', 'Budget', '$n$'] + [c for _, c in COLS]) + ' \\\\'
(HERE / 'table_main.tex').write_text('\\begin{tabular}{llr' + 'r' * len(COLS) + '}\n\\toprule\n' + header +
                                     '\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
print('ok', len(lines), 'macros')
