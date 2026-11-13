"""Label-based audit of Qwen3-Reranker-0.6B evidence extraction (evaluation only).

Gold evidence: LongHealth `answer_location` (character fractions per source
text) and QASPER annotated evidence paragraphs. MTOB has no evidence labels.
Gold labels are used here only to measure retrieval; they never entered the
pipeline that produced the saved evidence.
"""
import csv
import json
import math
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/da839/Query_Cart')
sys.path.insert(0, str(ROOT))
from experiments.run_figure4_contextual import load_cases  # noqa: E402

DATA = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_cartridges_figure4_datasets_20261110'
SYN = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/synthetic_threeway_min15_20261112/synthetic')
OUT = Path(__file__).resolve().parent
SNIPPET = 60


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def snippets(document, spans):
    out = []
    for a, b in spans:
        text = norm(document[a:b])
        if len(text) < 8:
            continue
        mid = len(text) // 2
        out.append(text[max(0, mid - SNIPPET // 2): mid + SNIPPET // 2])
    return out


def hits(text, gold):
    text = norm(text)
    return any(s in text for s in gold)


def longhealth_gold(root):
    data = json.loads(Path(root, 'longhealth', 'benchmark_v5.json').read_text())
    gold = {}
    for pid in [f'patient_{i:02d}' for i in range(1, 11)]:
        patient = data[pid]
        offsets, cursor = {}, 0
        for key in sorted(patient['texts']):
            offsets[key] = cursor
            cursor += len(patient['texts'][key]) + 2
        for q in patient['questions']:
            spans = []
            for key, loc in (q.get('answer_location') or {}).items():
                n = len(patient['texts'][key])
                for s, e in zip(loc['start'], loc['end']):
                    spans.append((offsets[key] + int(s * n), offsets[key] + int(math.ceil(e * n))))
            gold[f"{pid}-q{int(q['No']):02d}"] = spans
    return gold


def qasper_gold(root, cases):
    import pandas as pd
    frame = pd.read_parquet(Path(root, 'qasper', 'qasper-rewrite-gpt-4.1', 'data',
                                 'question-00000-of-00001.parquet'))
    gold = {}
    for index, row in frame.iterrows():
        cid = f"qasper-{index:03d}-{row['paper_id']}"
        if cid not in cases:
            continue
        document = cases[cid]['document']
        spans = []
        for paragraph in list(row['old_answer']['evidence']):
            at = document.find(paragraph)
            if at >= 0 and paragraph.strip():
                spans.append((at, at + len(paragraph)))
        gold[cid] = spans
    return gold


def lexical_score(query, text):
    stop = {"the", "a", "an", "of", "to", "in", "is", "are", "was", "were",
            "what", "which", "who", "where", "when", "how", "and", "or", "for"}
    counts = Counter(w for w in re.findall(r"[^\W_]+", query.casefold()) if len(w) > 1 and w not in stop)
    low = text.casefold()
    return sum(weight * low.count(word) for word, weight in counts.items())


def first_hit_rank(order, relevant):
    for rank, i in enumerate(order, 1):
        if relevant[i]:
            return rank
    return None


def main():
    from cartridges.structs import read_conversations
    rows, per_dataset = [], defaultdict(list)
    for dataset in ('longhealth', 'qasper'):
        cases = {c['id']: c for c in load_cases(dataset, DATA)}
        gold_spans = longhealth_gold(DATA) if dataset == 'longhealth' else qasper_gold(DATA, cases)
        for folder in sorted((SYN / dataset).iterdir()):
            source = folder / 'sources.json'
            if not source.exists() or folder.name not in gold_spans:
                continue
            case = cases[folder.name]
            gold = snippets(case['document'], gold_spans[folder.name])
            if not gold:
                continue
            record = json.loads(source.read_text())
            retrieval = record['retrieval']
            singles = retrieval['single_scores']
            chunk_text = [case['document'][a:b] for a, b in (s['span'] for s in singles)]
            relevant = [hits(t, gold) for t in chunk_text]
            if not any(relevant):
                continue  # gold falls between chunk boundaries after normalization
            n = len(singles)
            reranker = sorted(range(n), key=lambda i: (-singles[i]['score'], i))
            lexical = sorted(range(n), key=lambda i: (-lexical_score(case['query'], chunk_text[i]), i))
            groups = retrieval['selected_evidence']
            group_hit = [hits(g['text'], gold) for g in groups]
            specific = json.loads((folder / 'specific.json').read_text())['examples']
            spec_hit = [hits(ex['evidence']['text'], gold) for ex in specific]
            agnostic = read_conversations(str(folder / 'agnostic.targets.pkl'))[:15]
            agn_hit = [hits(c.system_prompt, gold) for c in agnostic]
            random_hit1 = sum(relevant) / n
            row = dict(dataset=dataset, case=folder.name, chunks=n, gold_chunks=sum(relevant),
                       reranker_rank=first_hit_rank(reranker, relevant),
                       lexical_rank=first_hit_rank(lexical, relevant),
                       random_hit1=random_hit1,
                       evidence_groups=len(groups), groups_with_gold=sum(group_hit),
                       first_group_has_gold=group_hit[0] if groups else False,
                       specific_trajectories=len(spec_hit), specific_with_gold=sum(spec_hit),
                       agnostic_trajectories=len(agn_hit), agnostic_with_gold=sum(agn_hit),
                       mean_reranker_score_gold=sum(singles[i]['score'] for i in range(n) if relevant[i]) / sum(relevant),
                       mean_reranker_score_other=(sum(singles[i]['score'] for i in range(n) if not relevant[i])
                                                  / max(1, n - sum(relevant))))
            rows.append(row)
            per_dataset[dataset].append(row)
    summary = {}
    for dataset, items in per_dataset.items():
        def hit(key, k):
            return sum(1 for r in items if r[key] is not None and r[key] <= k) / len(items)
        def random_hitk(k):
            # exact hypergeometric chance that a random k-subset contains a gold chunk
            total = 0.
            for r in items:
                n, g = r['chunks'], r['gold_chunks']
                miss = math.comb(n - g, min(k, n)) / math.comb(n, min(k, n)) if n - g >= min(k, n) else 0.
                total += 1 - miss
            return total / len(items)
        summary[dataset] = dict(
            requests=len(items), mean_chunks=sum(r['chunks'] for r in items) / len(items),
            mean_gold_chunks=sum(r['gold_chunks'] for r in items) / len(items),
            hit_at_k={m: {k: hit(f'{m}_rank', k) for k in (1, 3, 5)} for m in ('reranker', 'lexical')}
                     | {'random': {k: random_hitk(k) for k in (1, 3, 5)}},
            mrr={m: sum(1 / r[f'{m}_rank'] for r in items) / len(items) for m in ('reranker', 'lexical')},
            first_group_has_gold=sum(r['first_group_has_gold'] for r in items) / len(items),
            fraction_groups_with_gold=sum(r['groups_with_gold'] for r in items) / sum(r['evidence_groups'] for r in items),
            fraction_specific_trajectories_with_gold=sum(r['specific_with_gold'] for r in items)
                / sum(r['specific_trajectories'] for r in items),
            fraction_agnostic_trajectories_with_gold=sum(r['agnostic_with_gold'] for r in items)
                / sum(r['agnostic_trajectories'] for r in items),
            mean_score_gold_chunks=sum(r['mean_reranker_score_gold'] for r in items) / len(items),
            mean_score_other_chunks=sum(r['mean_reranker_score_other'] for r in items) / len(items))
    summary['_notes'] = dict(gold_match=f'central {SNIPPET}-character whitespace-normalized snippet of each gold span',
        mtob='no evidence annotation; not evaluated', gold_used_by_pipeline=False,
        excluded='requests whose gold could not be located in the document or chunks')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (OUT / 'per_request.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
