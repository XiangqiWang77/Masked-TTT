"""Snapshot of the OLMo-3 per-question synthesis (prompt v2): request status, candidate fate, token lengths."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/home/da839/Query_Cart')
from transformers import AutoTokenizer
from cartridges.structs import read_conversations
from contextualized.answer_guided_questions import parse_object, closed_book_question_rejection

O = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/olmo3_full_table_20261113')
tok = AutoTokenizer.from_pretrained('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/models/Olmo-3-7B-Instruct')
key = lambda s: ' '.join(s.casefold().split())
requests = json.loads((O / 'requests.json').read_text())
rows, fate, tokens = [], Counter(), []
for r in requests:
    root = O / 'synthetic' / r['dataset'] / r['case']
    status = 'complete' if (root / 'pair.json').exists() else 'shortfall' if (root / 'shortfall.json').exists() \
        else 'in progress' if (root / 'calls.jsonl').exists() else 'not started'
    spec = json.loads((root / 'specific.json').read_text())['examples'] if (root / 'specific.json').exists() else []
    state = json.loads((root / 'generation_state.json').read_text()) if (root / 'generation_state.json').exists() else {}
    calls = [json.loads(l) for l in open(root / 'calls.jsonl')] if (root / 'calls.jsonl').exists() else []
    answer_calls = [c for c in calls if c['instruction'].startswith('Read the evidence')]
    question_calls = [c for c in calls if not c['instruction'].startswith('Read the evidence')]
    # fate of every proposed question (re-applying the same admission checks; accepted = saved examples)
    seen = set()
    proposed = 0
    for c in question_calls:
        obj = parse_object(c['response']) if c['complete'] else None
        qs = obj.get('questions', []) if isinstance(obj, dict) else []
        if not qs:
            fate['no parsable question'] += 1; continue
        concl = c['payload']['conclusions']
        for v in qs:
            proposed += 1
            if not isinstance(v, dict) or not isinstance(v.get('question'), str):
                fate['malformed'] += 1; continue
            idx = v.get('answer_index')
            if type(idx) is not int or not 0 <= idx < len(concl):
                fate['malformed'] += 1; continue
            q = v['question'].strip()
            if key(q) in seen:
                fate['duplicate question'] += 1; continue
            seen.add(key(q))
            if closed_book_question_rejection(q, concl[idx]['answer'], []):
                fate['rejected: not closed-book'] += 1; continue
            fate['passed filters'] += 1
    empty_answers = sum('"answers": []' in c['response'] or '"answers":[]' in c['response'] for c in answer_calls)
    fate['answer step: no conclusion'] += empty_answers
    rows.append(dict(dataset=r['dataset'], case=r['case'], status=status, accepted=len(spec),
                     attempts=state.get('specific_attempt', 0), answer_calls=len(answer_calls), proposed=proposed))
    for arm in ('specific', 'agnostic'):
        p = root / f'{arm}.targets.pkl'
        if p.exists():
            for conv in read_conversations(str(p)):
                tokens.append(dict(dataset=r['dataset'], arm=arm,
                                   question=len(tok.encode(conv.messages[0].content, add_special_tokens=False)),
                                   answer=len(conv.messages[-1].token_ids)))
accepted_total = sum(x['accepted'] for x in rows)
fate['accepted (teacher answer complete)'] = accepted_total
fate['passed filters'] = max(0, fate['passed filters'] - accepted_total)   # remainder: incomplete teacher answer / over quota
fate['teacher answer incomplete / over quota'] = fate.pop('passed filters')
out = dict(requests=rows, fate=dict(fate), tokens=tokens)
(Path(__file__).with_name('snapshot.json')).write_text(json.dumps(out))
print(Counter(x['status'] for x in rows)); print(dict(fate)); print(len(tokens), 'trajectories')
