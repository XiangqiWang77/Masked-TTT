"""Choose singleton and joint-pair evidence with the frozen Qwen reranker.

Only the small reranker chooses evidence groups. The large generator later
receives these source groups and the real query to construct questions.
"""
import gc
from functools import partial
import itertools
import json
import math
from pathlib import Path
import re
import time

from experiments import build_synthetic_comparison as B
from experiments import qwen_reranker_candidates as R


VERSION = 'question-guided-single-pair-evidence-v1'
DEFAULT_POLICY = {'candidate_pool': 16, 'pair_candidates': 32, 'max_evidence': 32}


def policy_options(retrieval):
    result = {key: retrieval.get(key, value) for key, value in DEFAULT_POLICY.items()}
    if any(type(value) is not int or value < 1 for value in result.values()):
        raise ValueError('Evidence selection budgets must be positive integers')
    if result['candidate_pool'] > 16 or result['pair_candidates'] > 32 or result['max_evidence'] > 32:
        raise ValueError('This protocol permits at most 16 seed chunks, 32 pair forwards and 32 evidence groups')
    if result['max_evidence'] <= result['candidate_pool']:
        raise ValueError('Evidence budget must leave room for both retained singletons and pairs')
    decision_filter = retrieval.get('decision_filter', False)
    if type(decision_filter) is not bool:
        raise ValueError('decision_filter must be an explicit boolean')
    if decision_filter:
        minimum = retrieval.get('minimum_evidence_score', .5)
        if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                or not math.isfinite(minimum) or not .5 <= minimum < 1.):
            raise ValueError('minimum_evidence_score must be finite and in [0.5,1)')
        result.update(decision_filter=True, minimum_evidence_score=float(minimum))
    return result


def evidence_text(passages, indices):
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(passages) for i in indices):
        raise ValueError('Evidence must identify distinct canonical source chunks')
    return '\n\n'.join(f'[Source {i}]\n{passages[i]}' for i in sorted(indices))


def propose_pairs(query, passages, ranked_pool, maximum):
    """Bound expensive joint forwards using entities, adjacency and seed coverage."""
    from contextualized.data import STOP_WORDS, BRIDGE_STOP_WORDS
    query_words = {w for w in re.findall(r'\w+', query.casefold()) if w not in STOP_WORDS and len(w) > 1}
    entities = {}
    for i in ranked_pool:
        sentences = re.split(r'(?<=[.!?])\s+|\n+', passages[i])
        direct = [s for s in sentences if query_words.intersection(re.findall(r'\w+', s.casefold()))]
        selected = direct or sentences
        entities[i] = {word.casefold() for s in selected for word in re.findall(r'[^\W_]+', s)
            if (word[:1].isupper() or word.isdigit()) and word.casefold() not in BRIDGE_STOP_WORDS}
    rank = {i: j for j, i in enumerate(ranked_pool)}
    all_pairs = list(itertools.combinations(sorted(ranked_pool), 2))
    def priority(pair):
        a, b = pair; shared = entities[a] & entities[b]
        return (not bool(shared), b-a != 1, -len(shared), min(rank[a], rank[b]),
                max(rank[a], rank[b]), b-a, a, b)
    ordered = sorted(all_pairs, key=priority)
    chosen = []
    # Every ranked seed gets a proposed partner when the bounded pool permits.
    for seed in ranked_pool:
        candidate = next((pair for pair in ordered if seed in pair and pair not in chosen), None)
        if candidate is not None and len(chosen) < maximum: chosen.append(candidate)
    chosen += [pair for pair in ordered if pair not in chosen][:max(0, maximum-len(chosen))]
    return [{'source_indices': list(pair), 'shared_entities': sorted(entities[pair[0]] & entities[pair[1]]),
             'adjacent_sources': pair[1]-pair[0] == 1, 'proposal_uses_answer_labels': False} for pair in chosen]


def checked_score(value):
    if not isinstance(value, dict): raise ValueError('Reranker must return score and raw yes/no logits')
    for name in ('score', 'logit_no', 'logit_yes'):
        x = value.get(name)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('Nonfinite or missing reranker output')
    if not 0 <= value['score'] <= 1: raise ValueError('Reranker score must be in [0,1]')
    d = max(-700, min(700, value['logit_yes']-value['logit_no']))
    if abs(value['score']-1/(1+math.exp(-d))) > 2e-6:
        raise ValueError('Reranker score disagrees with its logits')
    return dict(value)


def craft_evidence(query, passages, spans, score_passages, policy=None, *, score_pairs=None):
    """Score every singleton and each proposed concatenated pair independently.

    ``score_pairs`` can use the same frozen model with a complementary-pair
    instruction. If omitted, the supplied scorer handles both calls.
    There are no extracted answers, synthetic questions or large-model calls.
    """
    policy = policy_options(policy or {})
    if not passages or len(passages) != len(spans): raise ValueError('All canonical passages and offsets are required')
    raw_singles = score_passages(query, passages)
    if len(raw_singles) != len(passages): raise ValueError('Reranker did not score all singleton passages')
    singles = [{'source_index': i, 'span': spans[i], **checked_score(value)} for i, value in enumerate(raw_singles)]
    pool = sorted(range(len(passages)), key=lambda i: (-singles[i]['score'], i))[:policy['candidate_pool']]
    proposals = propose_pairs(query, passages, pool, policy['pair_candidates'])
    pair_texts = [evidence_text(passages, row['source_indices']) for row in proposals]
    raw_pairs = (score_pairs or score_passages)(query, pair_texts) if pair_texts else []
    if len(raw_pairs) != len(proposals): raise ValueError('Reranker did not score every proposed joint pair')
    pairs = [{**proposal, **checked_score(value), 'score_source': 'joint_pair_model_forward'}
             for proposal, value in zip(proposals, raw_pairs)]
    selected_pairs = sorted(pairs, key=lambda row: (-row['score'], row['source_indices']))[:policy['max_evidence']-len(pool)]
    selected = []; decisions = []
    filtered = policy.get('decision_filter', False)
    def accept(indices, row, kind):
        if not filtered: return True
        reason = ('model_no_or_tie' if row['logit_yes'] <= row['logit_no'] else
                  'below_minimum_evidence_score' if row['score'] < policy['minimum_evidence_score'] else
                  'positive_reranker_decision')
        keep = reason == 'positive_reranker_decision'
        decisions.append({'source_indices': sorted(indices), 'kind': kind, 'retained': keep,
            'reason': reason, 'score': row['score'], 'logit_yes': row['logit_yes'], 'logit_no': row['logit_no']})
        return keep
    def append(indices, score, kind):
        indices = sorted(indices)
        selected.append({'id': ('source_' if kind == 'singleton' else 'pair_')+'_'.join(map(str, indices)),
            'kind': kind, 'source_indices': indices, 'text': evidence_text(passages, indices),
            'source_chunks': [{'id': str(i), 'source_index': i, 'span': spans[i], 'text': passages[i],
                              'reranker_score': singles[i]['score']} for i in indices],
            'reranker_score': score, 'semantic_quality_verified': False})
    # Interleave joint and local targets, avoiding a singleton-only prefix.
    for j in range(max(len(pool), len(selected_pairs))):
        if j < len(selected_pairs) and accept(selected_pairs[j]['source_indices'], selected_pairs[j], 'pair'):
            append(selected_pairs[j]['source_indices'], selected_pairs[j]['score'], 'pair')
        if j < len(pool) and accept([pool[j]], singles[pool[j]], 'singleton'):
            append([pool[j]], singles[pool[j]]['score'], 'singleton')
    result = {'single_scores': singles, 'ranked_candidate_pool': pool, 'pair_scores': pairs,
        'selected_evidence': selected, 'policy': policy,
        'selection': 'retain_top_singletons_and_rank_actual_joint_pair_forwards',
        'pair_scores_are_sums_of_singletons': False, 'pair_scores_prove_complementarity': False,
        'model_answer_generation_performed': False, 'benchmark_answers_used': False}
    if filtered:
        result.update(selection='positive_decisions_after_high_recall_singleton_and_joint_pair_ranking',
            final_group_decisions=decisions, minimum_evidence_score=policy['minimum_evidence_score'],
            candidate_pool_filtered_before_pair_proposal=False,
            requested_group_budget=policy['max_evidence'], retained_group_count=len(selected))
    return result


class FrozenReranker:
    """One GPU replica, used for all source groups before the 4B model loads."""
    def __init__(self, model_info, batch_size):
        self.model_info, self.batch_size = model_info, batch_size
        self.model = self.head = self.bias = self.tokenizer = None

    def __enter__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info['model'], padding_side='left',
            local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.no, self.yes = [self.tokenizer.convert_tokens_to_ids(s) for s in ('no', 'yes')]
        if self.no == self.yes or any(self.tokenizer.encode(s, add_special_tokens=False) != [i]
                                     for s, i in [('no', self.no), ('yes', self.yes)]):
            raise ValueError('Official reranker requires distinct single-token yes/no labels')
        self.model = AutoModelForCausalLM.from_pretrained(self.model_info['model'], torch_dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='flash_attention_2', local_files_only=True,
            trust_remote_code=False).eval().requires_grad_(False)
        self.head = self.model.lm_head.weight[[self.no, self.yes]]
        self.bias = self.model.lm_head.bias
        if self.bias is not None: self.bias = self.bias[[self.no, self.yes]]
        return self

    def to(self, device):
        """Park the decision model on CPU while the Cartridge is learning."""
        self.model.to(device)
        self.head = self.head.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self

    def score(self, query, passages, *, instruction=None):
        import torch
        from torch.nn import functional as F
        records = []
        for start in range(0, len(passages), self.batch_size):
            texts = passages[start:start+self.batch_size]
            selected_instruction = self.model_info['instruction'] if instruction is None else instruction
            rows = [self.tokenizer.encode(R.ranking_prompt(query, p, selected_instruction),
                                         add_special_tokens=False) for p in texts]
            if any(len(row) > min(self.model_info['max_length'], self.model.config.max_position_embeddings) for row in rows):
                raise ValueError('Reranker input exceeds the limit; source evidence is never truncated')
            with torch.inference_mode():
                batch = self.tokenizer.pad({'input_ids': rows}, padding=True, return_tensors='pt').to('cuda')
                positions = (batch.attention_mask.cumsum(-1)-1).clamp_min(0)
                hidden = self.model.model(**batch, position_ids=positions, use_cache=False).last_hidden_state[:, -1]
                logits = F.linear(hidden, self.head, self.bias).float()
                pairs = logits.cpu().tolist(); scores = logits.softmax(-1)[:, 1].cpu().tolist()
            records += [{'score': score, 'logit_no': pair[0], 'logit_yes': pair[1], 'input_tokens': len(row)}
                        for pair, score, row in zip(pairs, scores, rows)]
            del batch, positions, hidden, logits
        return records

    def __exit__(self, *_):
        import torch
        self.model = self.head = self.bias = self.tokenizer = None
        gc.collect(); torch.cuda.empty_cache()


def contract_for(case, spans, canonical, model_info, policy):
    return {**R.score_contract(case, spans, canonical, model_info),
            'evidence_policy': policy, 'evidence_protocol': VERSION}


def validate_evidence_record(record, case, expected_contract=None):
    contract = record.get('contract', {}); content = record.get('content', {})
    if (record.get('schema') != VERSION or record.get('complete') is not True
            or record.get('identity') != B.fingerprint(contract)
            or record.get('content_sha256') != B.fingerprint(content)
            or contract.get('case') != case['key'] or contract.get('query_sha256') != B.text_hash(case['query'])
            or contract.get('document_sha256') != B.text_hash(case['document'])
            or contract.get('evidence_protocol') != VERSION
            or contract.get('model', {}).get('repo') != R.REPO
            or contract.get('model', {}).get('revision') != R.REVISION
            or expected_contract is not None and contract != expected_contract):
        raise ValueError('Evidence cache has a different source, model or selection protocol')
    policy = policy_options(contract['evidence_policy']); spans = contract['source_spans']
    previous = -1
    if not isinstance(spans, list) or not spans: raise ValueError('Canonical spans are required')
    for span in spans:
        if (not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= len(case['document']) or span[0] <= previous):
            raise ValueError('Invalid canonical source position')
        previous = span[0]
    singles = content.get('single_scores', []); pairs = content.get('pair_scores', [])
    if len(singles) != len(spans) or len(pairs) > policy['pair_candidates']:
        raise ValueError('Missing singleton scores or excessive joint pair forwards')
    for i, row in enumerate(singles):
        checked_score(row)
        if row['source_index'] != i or row['span'] != spans[i]: raise ValueError('Singleton source positions changed')
    expected_pool = sorted(range(len(spans)), key=lambda i: (-singles[i]['score'], i))[:policy['candidate_pool']]
    if content.get('ranked_candidate_pool') != expected_pool: raise ValueError('The candidate pool is not reranker-ranked')
    pair_lookup = {}
    for row in pairs:
        checked_score(row); indices = tuple(row.get('source_indices', []))
        if (len(indices) != 2 or indices != tuple(sorted(set(indices))) or not set(indices).issubset(expected_pool)
                or indices in pair_lookup or row.get('score_source') != 'joint_pair_model_forward'):
            raise ValueError('Invalid or duplicated joint pair score')
        pair_lookup[indices] = row['score']
    passages = [case['document'][a:b] for a, b in spans]
    groups = content.get('selected_evidence', []); kept_singletons = []; kept_pairs = []; seen = set()
    if ((not groups and not policy.get('decision_filter', False)) or len(groups) > policy['max_evidence']):
        raise ValueError('Empty or excessive evidence groups')
    for group in groups:
        indices = group.get('source_indices', []); key = tuple(indices)
        if (not indices or indices != sorted(set(indices)) or not set(indices).issubset(expected_pool) or key in seen
                or group.get('text') != evidence_text(passages, indices)):
            raise ValueError('Selected evidence does not match original source text/order')
        expected_chunks = [{'id': str(i), 'source_index': i, 'span': spans[i], 'text': passages[i],
                            'reranker_score': singles[i]['score']} for i in indices]
        if group.get('source_chunks') != expected_chunks: raise ValueError('Selected evidence source provenance changed')
        if group.get('kind') == 'singleton' and len(indices) == 1:
            kept_singletons.append(indices[0]); value = singles[indices[0]]['score']
            expected_id = 'source_'+str(indices[0])
        elif group.get('kind') == 'pair' and key in pair_lookup:
            kept_pairs.append(key); value = pair_lookup[key]
            expected_id = 'pair_'+'_'.join(map(str, indices))
        else: raise ValueError('Selected group lacks an actual model forward')
        if group.get('id') != expected_id or group.get('reranker_score') != value:
            raise ValueError('Selected group ID/score differs from its model forward')
        seen.add(key)
    desired_pair_rows = sorted(pairs, key=lambda r: (-r['score'], r['source_indices']))[:policy['max_evidence']-len(expected_pool)]
    desired_singletons = expected_pool
    if policy.get('decision_filter', False):
        def accepted(row):
            return row['logit_yes'] > row['logit_no'] and row['score'] >= policy['minimum_evidence_score']
        desired_singletons = [i for i in expected_pool if accepted(singles[i])]
        desired_pair_rows = [row for row in desired_pair_rows if accepted(row)]
    desired_pairs = [tuple(row['source_indices']) for row in desired_pair_rows]
    if kept_singletons != desired_singletons or kept_pairs != desired_pairs:
        raise ValueError('Evidence selection dropped a protected singleton or misranked joint pairs')
    if content.get('benchmark_answers_used') is not False or content.get('model_answer_generation_performed') is not False:
        raise ValueError('This evidence stage cannot consume answer labels or generated answers')
    return groups


def prepare_question_guided_evidence(manifest, cases, output, stop_requested=None):
    """Return complete case→JSON paths; score singletons and pairs with one load."""
    from contextualized.data import read_indexed_case
    retrieval = manifest['retrieval']; model_info = R.model_contract(retrieval); policy = policy_options(retrieval)
    batch_size = retrieval.get('batch_size', 64)
    if type(batch_size) is not int or batch_size < 1: raise ValueError('Positive reranker batch size required')
    plan = B.read(manifest['full_plan']); index = {r['key']: r for r in B.read(plan['source_index'])}
    mapping = B.read(Path(plan['shared_scores'])/'case_to_score_file.json')
    output = Path(output); output.mkdir(parents=True, exist_ok=True); paths, missing = {}, []
    for item in cases:
        case = read_indexed_case(plan, index, item['key']); spans, _, canonical = R.canonical_chunks(case, plan, mapping)
        contract = contract_for(case, spans, canonical, model_info, policy); path = output/(case['key']+'.json')
        if path.exists():
            validate_evidence_record(B.read(path), case, contract); paths[case['key']] = str(path)
        else: missing.append(item)
    if not missing or stop_requested is not None and stop_requested(): return paths
    with FrozenReranker(model_info, batch_size) as scorer:
        for item in missing:
            if stop_requested is not None and stop_requested(): break
            case = read_indexed_case(plan, index, item['key']); spans, passages, canonical = R.canonical_chunks(case, plan, mapping)
            contract = contract_for(case, spans, canonical, model_info, policy); start = time.monotonic()
            content = craft_evidence(case['query'], passages, spans, scorer.score, policy,
                score_pairs=partial(scorer.score, instruction=model_info['pair_instruction']))
            record = {'schema': VERSION, 'complete': True, 'contract': contract, 'identity': B.fingerprint(contract),
                'content': content, 'content_sha256': B.fingerprint(content), 'wall_seconds': time.monotonic()-start,
                'small_model_generated_tokens': 0, 'large_model_used': False, 'semantic_quality_verified': False}
            validate_evidence_record(record, case, contract)
            path = output/(case['key']+'.json'); B.write(path, record); paths[case['key']] = str(path)
            print(json.dumps({'event': 'question_guided_evidence_ready', 'case': case['key'],
                'single_scores': len(content['single_scores']), 'joint_pair_scores': len(content['pair_scores']),
                'selected_groups': len(content['selected_evidence']), 'seconds': record['wall_seconds']}), flush=True)
    return paths


def original_contexts_from_evidence(case, tokenizer, chunk_tokens, evidence_record):
    """Keep the vanilla original context recipe unchanged beside the new E groups."""
    groups = validate_evidence_record(evidence_record, case); spans = evidence_record['contract']['source_spans']
    encoded = tokenizer(case['document'], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if ids != tokenizer.encode(case['document'], add_special_tokens=False): raise ValueError('Tokenizer offset mismatch')
    original = []
    for start in range(0, len(ids), chunk_tokens):
        end = min(start+chunk_tokens, len(ids)); char_span = [offsets[start][0], offsets[end-1][1]]
        original.append({'context_id': f'original_chunk_{len(original):04d}', 'kind': 'decoded_token_chunk',
            'text': tokenizer.decode(ids[start:end]), 'document_token_span': [start, end], 'document_char_span': char_span,
            'source_chunks': [{'source_index': i, 'span': span} for i, span in enumerate(spans)
                              if span[0] < char_span[1] and span[1] > char_span[0]]})
    scores = evidence_record['content']['single_scores']
    sources = [{'source_index': i, 'span': span, 'text': case['document'][span[0]:span[1]],
                'reranker_score': scores[i]['score']} for i, span in enumerate(spans)]
    return original, {'case': case['key'], 'document_sha256': B.text_hash(case['document']), 'source_chunks': sources,
        'contexts': original+[{'context_id': g['id'], 'kind': g['kind'], 'text': g['text'],
                              'source_chunks': g['source_chunks']} for g in groups],
        'retrieval': {'method': VERSION, 'selected_evidence_ids': [g['id'] for g in groups],
                     'evidence_identity': evidence_record['identity'], 'joint_pairs_scored_by_model': True,
                     'reference_answers_used': False},
        'source_facts_verified': False}
