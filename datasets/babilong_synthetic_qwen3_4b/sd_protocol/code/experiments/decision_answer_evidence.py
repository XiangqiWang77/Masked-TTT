"""Cached Decision API selection with the old Answer-Guided recall safeguards.

Only cached single-source decisions are consumed. Source-link groups are the
existing deterministic candidates, not API judgments of pair sufficiency.
No model, network request, digest check, answer label, or training is needed.
"""
import math
import re
import time

from contextualized import answer_guided_questions as A
from contextualized.data import STOP_WORDS, BRIDGE_STOP_WORDS


VERSION = 'cached-decision-answer-evidence-v3'


def prioritize_decision_clusters(query, candidates, clusters, *, max_clusters, max_chunks):
    """Keep query-anchored evidence without filling the budget with known drops.

    The high-recall pool is still available when forming a cluster, so bridge
    chunks can accompany an anchored chunk.  A cluster containing *only*
    chunks rejected by the Decision API and with zero lexical query support is
    not itself a useful Answer-Guided exercise, however.  Previously those
    clusters were appended merely to fill ``max_clusters``; with eight question
    paraphrases per answer this allowed unrelated targets to occupy half of a
    case's training budget.
    """
    names = {word.casefold() for word in re.findall(r'[^\W_]+', query)
             if word[:1].isupper() and word.casefold() not in STOP_WORDS}
    named = [row for row in candidates
             if names.intersection(re.findall(r'\w+', row['text'].casefold()))]
    kept = sorted((row for row in candidates if row['decision_api_accepted']),
                  key=lambda row: (-(row['decision_api_probability_yes'] or 0), row['source_index']))
    priority = list(dict.fromkeys(row['source_index'] for row in named+kept))
    by_index = {row['source_index']: row for row in candidates}
    result, seen = [], set()

    def append(chunks, origin):
        chunks = sorted(chunks, key=lambda row: row['source_index'])
        identity = tuple(row['source_index'] for row in chunks)
        if not identity or identity in seen or len(result) >= max_clusters:
            return
        seen.add(identity)
        result.append({'id': 'cluster_'+'_'.join(row['id'] for row in chunks),
            'origin': origin, 'source_chunks': chunks,
            'text': '\n\n'.join(f'[Source {row["id"]}]\n{row["text"]}' for row in chunks)})

    # Start with API-selected query entities, then retain broader recall groups.
    named_indices = {row['source_index'] for row in named}
    direct_kept = [row for row in kept if row['source_index'] in named_indices]
    for start in range(0, len(direct_kept), max_chunks):
        append(direct_kept[start:start+max_chunks], 'query_entity_and_cached_api')
    for start in range(0, len(kept), max_chunks):
        append(kept[start:start+max_chunks], 'cached_decision_api_recall')
    for start in range(0, len(named), max_chunks):
        append(named[start:start+max_chunks], 'query_entity_recall')
    for index in priority[:max_chunks]:
        append([by_index[index]], 'query_or_decision_singleton')
    rejected_unanchored = 0
    for cluster in clusters:
        chunks = cluster['source_chunks']
        anchored = any(chunk.get('decision_api_accepted') is True
                       or float(chunk.get('lexical_score') or 0) > 0
                       for chunk in chunks)
        if anchored:
            append(chunks, cluster['origin'])
        else:
            rejected_unanchored += 1

    # Every exercise must retain the query entity's compact cross-chunk core.
    # Otherwise a valid but superseded singleton can teach the opposite of a
    # later fact (the exact fragmentation failure this dataset is intended to
    # remove).  Rank only already-retrieved candidates; this performs no new
    # model call and reads no answer label.  Always preserve the latest named
    # occurrence because BABILong state questions are order-sensitive.
    ranked_core = sorted(named or kept or [row for row in candidates
                                           if float(row.get('lexical_score') or 0) > 0],
        key=lambda row: (
            row.get('decision_api_accepted') is not True,
            -(row.get('decision_api_probability_yes') or 0),
            -float(row.get('lexical_score') or 0),
            row['source_index']))
    core = ranked_core[:max_chunks]
    if ranked_core:
        latest = max(ranked_core, key=lambda row: row['source_index'])
        if latest not in core:
            core[-1] = latest
    core_indices = {row['source_index'] for row in core}
    augmented, augmented_seen = [], set()
    for group in result:
        extras = [row for row in group['source_chunks']
                  if row['source_index'] not in core_indices]
        merged = sorted(core + extras[:max(0, max_chunks-len(core))],
                        key=lambda row: row['source_index'])
        identity = tuple(row['source_index'] for row in merged)
        if not identity or identity in augmented_seen:
            continue
        augmented_seen.add(identity)
        augmented.append({'id': 'cluster_'+'_'.join(row['id'] for row in merged),
            'origin': group['origin']+'+query_complete_core',
            'source_chunks': merged,
            'text': '\n\n'.join(f'[Source {row["id"]}]\n{row["text"]}' for row in merged),
            'query_complete_core_source_indices': sorted(core_indices)})
    return augmented, priority, rejected_unanchored


def canonical_sources(case, score_record):
    """Read original slices and decisions using plain identity/shape checks."""
    if (score_record.get('schema') != 'masked-ttt-source-scores-v1'
            or score_record.get('complete') is not True
            or score_record.get('case_key') != case['key']
            or score_record.get('question') != case['query']
            or score_record.get('reference_answers_used') is not False):
        raise ValueError('Cached Decision API record does not match this label-free case')
    spans, records = score_record.get('source_spans'), score_record.get('records')
    if not isinstance(spans, list) or not spans or not isinstance(records, list) or len(spans) != len(records):
        raise ValueError('Source spans and cached decisions must be complete')
    normalized, previous = [], -1
    for i, (span, record) in enumerate(zip(spans, records)):
        if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(case['document']) or span[0] <= previous
                or not isinstance(record, dict) or record.get('chunk') != i
                or type(record.get('accepted')) is not bool):
            raise ValueError('Invalid original source span or cached decision')
        previous = span[0]; normalized.append(list(span))
    adjustments = []
    for i in range(1, len(normalized)):
        prior, current = normalized[i-1], normalized[i]
        if current[0] < prior[1]:
            if not prior[0] < current[0] < prior[1] <= current[1]:
                raise ValueError('Contained source spans cannot be reconciled')
            raw_span = list(prior); prior[1] = current[0]
            adjustments.append({'source_index': i-1, 'raw_span': raw_span,
                'normalized_span': list(prior), 'overlap_assigned_to_source_index': i,
                'unique_source_text_removed': False})
    changed = {row['source_index']: row['raw_span'] for row in adjustments}
    sources = []
    for i, (span, record) in enumerate(zip(normalized, records)):
        probability = record.get('decision_probabilities', {}).get('yes')
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            probability = None
        sources.append({'id': str(i), 'source_index': i, 'span': span,
            **({'raw_span': changed[i]} if i in changed else {}),
            'text': case['document'][span[0]:span[1]],
            'decision_api_accepted': record['accepted'],
            'decision_api_probability_yes': probability,
            'decision_api_reason': record.get('reason'),
            'decision_api_choice': record.get('decision_choice')})
    return sources, adjustments


def lexical_recall(sources, query, candidate_count):
    """Keep the existing TF-IDF, entity-bridge and neighbor seed algorithm."""
    chunks = [row['text'] for row in sources]
    tokenize_words = lambda text: re.findall(r'[\w]+', text.casefold())
    terms = sorted({w for w in tokenize_words(query) if w not in STOP_WORDS and len(w) > 1})
    words = [tokenize_words(text) for text in chunks]
    frequency = {term: sum(term in part for part in words) for term in terms}
    scores = []
    for part in words:
        counts = {term: part.count(term) for term in terms}
        value = sum((1+math.log(count))*math.log(1+(len(words)+1)/(frequency[term]+1))
                    for term, count in counts.items() if count)
        scores.append(value/math.sqrt(max(1, len(part))))
    rank = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    positive = [i for i in rank if scores[i] > 0]
    bridge = set()
    for chunk in chunks:
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', chunk):
            if not set(tokenize_words(sentence)).intersection(terms):
                continue
            for raw in re.findall(r'[^\W_]+', sentence, flags=re.UNICODE):
                word = raw.casefold()
                if word in terms or word in BRIDGE_STOP_WORDS:
                    continue
                if len(word) > 2 or (len(word) == 1 and raw.isupper() and raw not in {'A', 'I'}):
                    bridge.add(word)
    bridge_frequency = {term: sum(term in part for part in words) for term in bridge}
    bridge = set(sorted(bridge, key=lambda x: (bridge_frequency[x], x))[:32])
    bridge_scores = [sum(math.log(1+(len(words)+1)/(bridge_frequency[term]+1))
                         for term in bridge if term in part)/math.sqrt(max(1, len(part)))
                     for part in words]
    bridge_rank = sorted(range(len(chunks)), key=lambda i: (-bridge_scores[i], i))
    seeds = positive[:max(1, (candidate_count+2)//3)] or [len(chunks)//2]
    chosen = list(seeds)
    for i in bridge_rank:
        if len(chosen) >= min(candidate_count, len(seeds)+max(1, candidate_count//3)):
            break
        if bridge_scores[i] > 0 and i not in chosen:
            chosen.append(i)
    bridge_seeds = [i for i in chosen if i not in seeds]
    for center in seeds+bridge_seeds:
        for neighbor in (center-1, center+1):
            if 0 <= neighbor < len(chunks) and neighbor not in chosen and len(chosen) < candidate_count:
                chosen.append(neighbor)
    for i in rank:
        if len(chosen) >= min(candidate_count, len(chunks)):
            break
        if i not in chosen:
            chosen.append(i)
    chosen.sort()
    support_rank = chosen+[i for i in bridge_rank if bridge_scores[i] > 0 and i not in chosen]
    support_rank += [i for i in rank if i not in support_rank]
    for i, source in enumerate(sources):
        source.update(lexical_score=scores[i], bridge_score=bridge_scores[i])
    return {'seed_candidate_source_indices': chosen, 'query_terms': terms,
            'query_anchor_source_indices': positive, 'bridge_terms': sorted(bridge),
            'bridge_seed_indices': bridge_seeds, 'support_rank': support_rank}


def pack_sources(sources, rank, tokenizer, support_tokens, token_lengths):
    """Use the old support-view budget accounting, retaining source order."""
    selected, tokens = [], 0
    for i in rank:
        if i in selected:
            continue
        size = token_lengths[i]
        if not size or size > support_tokens or tokens+size > support_tokens:
            continue
        trial = sorted(selected+[i])
        actual = len(tokenizer.encode('\n\n'.join(sources[j]['text'] for j in trial),
                                      add_special_tokens=False))
        if actual <= support_tokens:
            selected, tokens = trial, actual
    return selected, tokens


async def prepare_evidence(case, score_record, tokenizer, *, candidate_pool=16,
                           max_clusters=32, max_chunks=8, support_tokens=32768):
    """Return source_catalog, clusters and audit for the unchanged two-stage A.

    Cached KEEP decisions prioritize spare capacity. They cannot remove the
    existing lexical seed pool or its supported query anchors/entity bridges.
    A zero-KEEP case therefore keeps its previous recall safeguards. The same
    finite token/group budgets still apply and all omissions are reported.
    """
    if any(type(x) is not int or x < 1 for x in
           (candidate_pool, max_clusters, max_chunks, support_tokens)):
        raise ValueError('Evidence budgets must be positive integers')
    started = time.monotonic()
    sources, adjustments = canonical_sources(case, score_record)
    recall = lexical_recall(sources, case['query'], candidate_pool)
    token_lengths = [len(tokenizer.encode(row['text'], add_special_tokens=False)) for row in sources]
    old_selected, _ = pack_sources(sources, recall['support_rank'], tokenizer, support_tokens, token_lengths)
    seeds = recall['seed_candidate_source_indices']
    if not old_selected or not set(seeds).issubset(old_selected):
        raise ValueError('Support budget cannot retain the existing lexical seed pool')
    protected = [i for i in old_selected if i in seeds or sources[i]['lexical_score'] > 0
                 or sources[i]['bridge_score'] > 0]
    kept = sorted((i for i, row in enumerate(sources) if row['decision_api_accepted']),
                  key=lambda i: (-(sources[i]['decision_api_probability_yes'] or 0), i))
    priority = list(dict.fromkeys(protected+kept+old_selected+recall['support_rank']))
    selected, tokens = pack_sources(sources, priority, tokenizer, support_tokens, token_lengths)
    if not set(protected).issubset(selected):
        raise ValueError('Decision-prioritized support cannot retain protected recall sources')
    seed_rank = {i: rank for rank, i in enumerate(seeds)}
    for row in sources:
        row['seed_rank'] = seed_rank.get(row['source_index'])
    candidates = [sources[i] for i in selected]
    clustered = await A.propose_evidence_clusters(None, case['query'], candidates,
        max_clusters=max_clusters, max_chunks_per_cluster=max_chunks, use_model_proposals=False)
    old_clusters = clustered['clusters']
    clustered['clusters'], priority, rejected_unanchored = prioritize_decision_clusters(case['query'], candidates,
        old_clusters, max_clusters=max_clusters, max_chunks=max_chunks)
    clustered['audit'].update(priority_source_indices=priority,
        priority_policy='explicit_query_entities_then_cached_api_keep_before_generic_lexical_pairs',
        legacy_cluster_count=len(old_clusters), final_cluster_count=len(clustered['clusters']),
        rejected_unanchored_legacy_clusters=rejected_unanchored,
        exercise_admission=('at least one Decision-API KEEP or positive lexical-query anchor; '
                            'unselected bridge chunks may accompany an anchored chunk'),
        query_complete_core_added_to_every_exercise=True,
        query_complete_core_source_indices=(
            clustered['clusters'][0].get('query_complete_core_source_indices', [])
            if clustered['clusters'] else []),
        fragmentation_policy=(
            'every local focus retains the same retrieved query-entity core so later facts and '
            'cross-chunk relations cannot be dropped by singleton exercise construction'))
    used = sorted({chunk['source_index'] for group in clustered['clusters'] for chunk in group['source_chunks']})
    requests = score_record.get('requests', [])
    retrieval = {'method': VERSION, 'cached_decision_api_used': True, 'new_decision_api_requests': 0,
        'cached_decision_models': sorted({row['model'] for row in requests if isinstance(row, dict)
                                          and isinstance(row.get('model'), str)}),
        'cached_decision_policies': sorted({row['policy_version'] for row in requests if isinstance(row, dict)
                                            and isinstance(row.get('policy_version'), str)}),
        'source_chunks': len(sources), 'candidate_source_indices': selected,
        'seed_candidate_source_indices': seeds, 'protected_recall_source_indices': protected,
        'old_lexical_support_source_indices': old_selected,
        'decision_keep_source_indices': sorted(kept), 'zero_decision_keep': not kept,
        'decision_keep_excluded_by_budget': sorted(set(kept)-set(selected)),
        'decision_drop_retained_for_recall': [i for i in selected if not sources[i]['decision_api_accepted']],
        'query_terms': recall['query_terms'], 'bridge_terms': recall['bridge_terms'],
        'bridge_seed_indices': recall['bridge_seed_indices'],
        'support_tokens': tokens, 'support_token_limit': support_tokens,
        'support_pool_used_as_candidates': True, 'pair_synergy_api_judged': False,
        'reference_answers_used': False}
    audit = {'version': VERSION, 'case': case['key'], 'query': case['query'],
        'retrieval': retrieval, 'cluster_construction': clustered['audit'],
        'cluster_count': len(clustered['clusters']), 'cluster_source_indices': used,
        'candidate_sources_not_used_under_cluster_budget': sorted(set(selected)-set(used)),
        'wall_seconds': time.monotonic()-started, 'generation_model_calls': 0,
        'source_facts_verified': False, 'pair_synergy_api_judged': False,
        'benchmark_answer_values_used': False, 'training_performed': False}
    return {'source_catalog': {'case': case['key'], 'query': case['query'],
        'source_chunks': sources, 'candidate_source_indices': selected, 'retrieval': retrieval,
        'source_boundary_adjustments': adjustments, 'source_facts_verified': False},
        'clusters': clustered['clusters'], 'audit': audit}
