"""Answer-first dataset crafting using frozen text generation only.

Each ordered evidence cluster gets at most one extraction and one question call.
Literal source checks are provenance checks; no semantic correctness is claimed.
"""
import json
from itertools import combinations
import re
import time

from .question_format import query_request_rejection


VERSION = 'answer-guided-dataset-v2'
_REQUEST_START = re.compile(r'^\s*(?:who|what|where|when|why|how|which|is|are|was|were|does|do|did|can|'
                            r'could|should|would|name|identify|state|list|describe|explain|determine|trace|'
                            r'compare|summari[sz]e)\b', re.IGNORECASE)
_QUERY_STOP = {'a', 'an', 'and', 'are', 'did', 'do', 'does', 'for', 'from', 'how', 'in', 'is', 'of',
               'on', 'or', 'the', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'why',
               'before', 'after'}
_BRIDGE_STOP = _QUERY_STOP | {'here', 'there', 'somewhere', 'something', 'someone'}
_TEMPORAL_WORDS = {'after', 'before', 'current', 'currently', 'earlier', 'final', 'first', 'just',
                   'following', 'last', 'later', 'latest', 'next', 'now', 'previous', 'previously',
                   'prior', 'recent', 'recently', 'subsequent', 'subsequently', 'then'}
_DEICTIC = re.compile(r'\b(?:evidence|statement|stated|sentence|text|passage|source|quote|record|'
                       r'(?:given|provided|available) information|'
                       r'information (?:given|provided|available))\b', re.IGNORECASE)
_ACTION_REQUEST = re.compile(r'\b(?:what action|what did [^?]+ do)\b', re.IGNORECASE)
_WH_REQUEST = re.compile(r'^\s*(?:who|what|where|when|which|how|why)\b', re.IGNORECASE)
_YES_NO_ANSWER = re.compile(r'^\s*(?:yes|no)\b', re.IGNORECASE)
_UNSUPPORTED_ASSUMPTION = re.compile(r'\b(?:assume|assuming|presuming|suppose|supposing)\b',
                                     re.IGNORECASE)
_NO_INFORMATION_ANSWER = re.compile(
    r'\b(?:evidence|text|passage|source|provided information)\b[^.]{0,100}'
    r'\b(?:does not|doesn\'t|do not|no)\b[^.]{0,80}'
    r'\b(?:provide|mention|contain|include|information|detail)\w*\b', re.IGNORECASE)


def parse_object_with_status(text):
    if not isinstance(text, str): return None, 'unknown'
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(text)
        return (value, 'parsed') if isinstance(value, dict) else (None, 'unknown')
    except ValueError:
        try:
            value = json.loads(text, strict=False)
            if isinstance(value, dict): return value, 'parsed_with_json_control_characters'
        except ValueError:
            pass
        # Some frozen decoders render a quoted excerpt as [""sentence""].
        # Normalize only the invalid doubled delimiters at array-item edges;
        # the raw output remains in the audit and no second model call occurs.
        repaired = re.sub(r'([\[,]\s*)""', r'\1"', text)
        repaired = re.sub(r'""(\s*[,\]])', r'"\1', repaired)
        candidates = []
        if repaired != text: candidates.append((repaired, 'parsed_after_doubled_quote_normalization'))
        # A few deterministic local decodes append exactly one unmatched
        # closing brace. Accept it only when deleting that brace yields a
        # complete JSON object; preserve the original bytes in the audit.
        for value, status in [(text, 'parsed_after_trailing_brace_normalization'), *candidates]:
            if value.endswith('}}'):
                candidates.append((value[:-1], status if value != text else
                                   'parsed_after_trailing_brace_normalization'))
        for candidate, status in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict): return value, status
            except ValueError:
                pass
        return None, 'unknown'


def parse_object(text):
    return parse_object_with_status(text)[0]


def literal_quotes(quotes, source_chunks):
    """Locate every quote inside one concrete source chunk of this cluster."""
    if not isinstance(quotes, list) or not quotes: return None
    checked = []
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip(): return None
        surface = quote.strip()
        candidates = [surface]
        if surface.endswith((',', ';')): candidates.append(surface[:-1].rstrip())
        wrappers = {'"': '"', "'": "'", '“': '”', '‘': '’'}
        for candidate in list(candidates):
            if len(candidate) > 2 and wrappers.get(candidate[0]) == candidate[-1]:
                candidates.append(candidate[1:-1])
        matches, matched_quote = [], None
        for candidate in candidates:
            for chunk in source_chunks:
                text = chunk['text']; offset = 0
                while (start := text.find(candidate, offset)) >= 0:
                    span = chunk.get('span'); matched_quote = candidate
                    matches.append({'source_index': chunk.get('source_index'), 'quote': candidate,
                        'chunk_start': start, 'chunk_end': start+len(candidate),
                        'document_start': span[0]+start if isinstance(span, list) and len(span) == 2 else None,
                        'document_end': span[0]+start+len(candidate) if isinstance(span, list) and len(span) == 2 else None})
                    offset = start + max(1, len(candidate))
            if matches: break
        if not matches: return None
        checked.append({'quote': matched_quote, 'reported_quote': quote, 'matches': matches,
                        'literal_occurrence_count': len(matches)})
    return checked


def focused_evidence_view(evidence, query, *, simple_two_stage=False):
    """Keep query-bearing sentences in anchor chunks before answer extraction.

    Bridge chunks without a direct query term remain intact. This is a
    deterministic input view; stored evidence and provenance are unchanged.
    """
    terms = {word for word in re.findall(r'\w+', query.casefold())
             if len(word) > 1 and word not in _QUERY_STOP}
    prepared, bridge_terms = [], set()
    for chunk in evidence['source_chunks']:
        sentences = [match.group(0).strip() for match in
                     re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', chunk['text']) if match.group(0).strip()]
        direct = [sentence for sentence in sentences
                  if terms.intersection(re.findall(r'\w+', sentence.casefold()))]
        for sentence in direct:
            bridge_terms.update(word for word in re.findall(r'\w+', sentence.casefold())
                                if len(word) > 2 and word not in terms and word not in _BRIDGE_STOP)
        prepared.append((chunk, sentences, direct))
    pieces = []
    for chunk, sentences, direct in prepared:
        relevant = direct or [sentence for sentence in sentences
                              if bridge_terms.intersection(re.findall(r'\w+', sentence.casefold()))]
        separator = '\n\n' if simple_two_stage else ' '
        text = separator.join(relevant) if relevant else chunk['text']
        source_id = chunk.get('id', chunk.get('source_index'))
        pieces.append({'source_id': source_id, 'source_index': chunk.get('source_index'), 'text': text,
                       'query_anchor_excerpt': bool(relevant)})
        if simple_two_stage:
            pieces[-1].update(source_segments=relevant or [chunk['text']],
                              segments_may_have_source_gaps=True)
    return {'id': evidence['id'], 'text': '\n\n'.join(
        f'[Source {piece["source_id"]}]\n{piece["text"]}' for piece in pieces), 'source_excerpts': pieces}


def direct_answer_leakage(question, answer):
    """Catch whole answer phrases, avoiding substring and yes/no-format matches.

    This is a conservative lexical guard, not a semantic leakage verifier.
    Very short/boolean labels are too ambiguous for a bare occurrence to reject.
    """
    words = lambda text: re.findall(r'\w+', text.casefold())
    target, request = words(answer), words(question)
    if not target: return None
    if len(target) == 1 and (target[0] in {'yes', 'no', 'none', 'unknown', 'nothing'}
                             or (len(target[0]) < 3 and not target[0].isdigit())):
        return None
    if any(request[i:i+len(target)] == target for i in range(len(request)-len(target)+1)):
        return 'literal_answer_phrase_in_question'
    content = [word for word in target if word not in _QUERY_STOP | {'at', 'be', 'been', 'by', 'has',
               'have', 'had', 'into', 'onto', 'with'}]
    if len(content) == 1 and len(content[0]) >= 3 and content[0] in request:
        return 'literal_answer_value_in_question'
    return None


def unsupported_question_claim(question, answer, query=''):
    """Reject relations introduced by the question rather than its target answer."""
    request = set(re.findall(r'\w+', question.casefold()))
    target = set(re.findall(r'\w+', answer.casefold()))
    query_words = set(re.findall(r'\w+', query.casefold()))
    if (request & _TEMPORAL_WORDS) - target and not query_words.intersection(_TEMPORAL_WORDS):
        return 'unsupported_temporal_qualifier'
    if _DEICTIC.search(question): return 'evidence_deictic_question'
    if _ACTION_REQUEST.search(question):
        action = any(word.endswith(('ed', 'ing')) for word in target) or bool(
            target & {'gave', 'give', 'go', 'goes', 'went', 'move', 'moves', 'moved', 'take', 'takes', 'took'})
        if not action: return 'question_answer_type_mismatch'
    return None


def noninformative_answer(answer):
    """Reject meta-observations that teach no document fact.

    This is a narrow format contract, not a semantic verifier.  Uncertainty
    about a stated alternative remains useful; only claims that the supplied
    material contains no relevant information are removed.
    """
    return 'no_query_relevant_fact_in_answer' if _NO_INFORMATION_ANSWER.search(answer or '') else None


def closed_book_question_rejection(question, answer, supporting_quotes):
    """Enforce the two-stage closed-book interface without another model call."""
    reason = query_request_rejection(question)
    if reason is not None:
        return reason
    if '?' not in question and not _REQUEST_START.search(question):
        return 'not_a_question_or_request'
    if _DEICTIC.search(question):
        return 'evidence_deictic_question'
    if _UNSUPPORTED_ASSUMPTION.search(question):
        return 'question_adds_an_assumption'
    if _YES_NO_ANSWER.match(answer or '') and _WH_REQUEST.match(question or ''):
        return 'question_answer_type_mismatch'
    reason = direct_answer_leakage(question, answer)
    if reason is not None:
        return reason
    question_words = set(re.findall(r'\w+', question.casefold()))
    for quote in supporting_quotes:
        quote_words = [word for word in re.findall(r'\w+', quote.casefold())
                       if word not in _QUERY_STOP and len(word) > 2]
        if (len(quote_words) >= 4
                and sum(word in question_words for word in quote_words) / len(quote_words) >= .8):
            return 'supporting_fact_revealed_in_question'
    return None


async def propose_evidence_clusters(engine, query, source_units, *, max_clusters=32,
                                    max_chunks_per_cluster=8, cluster_tokens=4096, call_model=None,
                                    use_model_proposals=True):
    """Select source-ID groups before answer extraction, without making a goal checklist.

    Query-anchored lexical hits are never discarded within the cluster budget.
    Their singleton and pair clusters are deterministic recall safeguards; the
    frozen model can additionally connect one of those anchors to bridge text.
    ``use_model_proposals=False`` runs only the existing deterministic groups;
    it makes no model call and does not invent a model response.
    """
    if not isinstance(query, str) or not query.strip(): raise ValueError('A nonempty real query is required')
    if any(type(n) is not int or n < 1 for n in (max_clusters, max_chunks_per_cluster, cluster_tokens)):
        raise ValueError('Cluster budgets must be positive integers')
    units = json.loads(json.dumps(list(source_units), ensure_ascii=False, allow_nan=False))
    by_id, source_order = {}, {}
    query_terms = {word for word in re.findall(r'\w+', query.casefold())
                   if len(word) > 1 and word not in _QUERY_STOP}
    for unit in units:
        if (not isinstance(unit, dict) or not isinstance(unit.get('id'), str) or not unit['id'].strip()
                or not isinstance(unit.get('text'), str) or not unit['text'].strip()
                or not isinstance(unit.get('source_index'), int) or not isinstance(unit.get('span'), list)
                or len(unit['span']) != 2 or not all(type(value) is int for value in unit['span'])
                or not 0 <= unit['span'][0] < unit['span'][1]
                or unit['span'][1]-unit['span'][0] != len(unit['text'])):
            raise ValueError('Each source unit needs id, text, integer source_index, and span')
        if unit['id'] in by_id: raise ValueError('Source-unit IDs must be unique')
        words = set(re.findall(r'\w+', unit['text'].casefold()))
        score = unit.get('lexical_score')
        lexical_anchor = isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0
        unit['query_anchor'] = bool(lexical_anchor or not query_terms or query_terms.intersection(words))
        source_order[unit['id']] = len(source_order); by_id[unit['id']] = unit
    if not units: raise ValueError('At least one source unit is required')
    if not any(unit['query_anchor'] for unit in units):
        fallback = min((unit for unit in units if type(unit.get('seed_rank')) is int),
                       key=lambda unit: unit['seed_rank'], default=units[0])
        fallback['query_anchor'] = True
    call_model = call_model or frozen_text_call
    instruction = (
        'Locate compact evidence clusters for crafting study data about the real query. Return JSON only: '
        '{"clusters":[{"source_ids":["id1"]},{"source_ids":["id1","id2"]}]}. '
        'Each cluster must support one concrete answer, prerequisite, or intermediate conclusion useful for solving '
        'the query. The program already preserves every source marked query_anchor; focus on complementary multi-source '
        'clusters that connect an anchor to a necessary bridge. Every cluster must include at least one query_anchor. '
        'Keep every cluster minimal, use only supplied IDs, do not answer the query, and do not '
        f'return more than {max_clusters} clusters or {max_chunks_per_cluster} sources per cluster.')
    started = time.monotonic()
    response, parsed, parse_status = None, None, 'not_requested'
    if use_model_proposals:
        response = await call_model(engine, instruction, {'query': query, 'sources': units}, max_tokens=cluster_tokens)
        if not isinstance(response, dict): response = {'response_count': 0, 'raw_output': None,
                                                        'transport_result_type': type(response).__name__}
        parsed, parse_status = (parse_object_with_status(response.get('raw_output'))
                                if response.get('response_count') == 1 else (None, 'unknown'))
    raw = parsed.get('clusters') if isinstance(parsed, dict) else None
    clusters, rejected, seen = [], [], set()

    def add_cluster(ids, origin, index=None):
        """Materialize a validated cluster while preserving source order."""
        ids = list(dict.fromkeys(str(value) for value in ids))
        reason = None
        if not ids or any(value not in by_id for value in ids): reason = 'unknown_source_id'
        else: ids.sort(key=source_order.get)
        if reason is None:
            if len(ids) > max_chunks_per_cluster: reason = 'cluster_chunk_budget_exceeded'
            elif not any(by_id[value]['query_anchor'] for value in ids): reason = 'missing_query_anchor'
            elif tuple(ids) in seen: reason = 'duplicate_cluster'
            elif len(clusters) >= max_clusters: reason = 'cluster_count_budget_exceeded'
        if reason:
            if origin == 'model_proposed':
                rejected.append({'index': index, 'source_ids': ids, 'reason': reason})
            return False
        seen.add(tuple(ids)); chunks = [by_id[value] for value in ids]
        clusters.append({'id': 'cluster_' + '_'.join(ids), 'origin': origin,
            'text': '\n\n'.join(f'[Source {chunk["id"]}]\n{chunk["text"]}' for chunk in chunks),
            'source_chunks': chunks})
        return True

    anchored = [unit['id'] for unit in units if unit['query_anchor']]

    def anchor_link_terms(unit):
        """Named entities/numerals in the query-bearing sentence link facts across chunks."""
        sentences = [match.group(0) for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', unit['text'])]
        direct = [sentence for sentence in sentences
                  if query_terms.intersection(re.findall(r'\w+', sentence.casefold()))]
        terms = set()
        for sentence in direct:
            for raw in re.findall(r'[^\W_]+', sentence, flags=re.UNICODE):
                word = raw.casefold()
                if word in _BRIDGE_STOP or (word in query_terms and not raw[:1].isupper()): continue
                if raw[:1].isupper() or raw.isdigit(): terms.add(word)
        return terms

    link_terms = {source_id: anchor_link_terms(by_id[source_id]) for source_id in anchored}
    ranked_anchors = sorted(anchored, key=lambda source_id: (
        -(float(by_id[source_id].get('lexical_score', 0) or 0)
          + float(by_id[source_id].get('bridge_score', 0) or 0)),
        (by_id[source_id].get('seed_rank') if type(by_id[source_id].get('seed_rank')) is int else 10**9),
        source_order[source_id]))
    # Query-bearing chunks that mention the same entity form a deterministic
    # chain. Add the nearest preceding/following occurrence of that entity for
    # each anchor: this recovers dispersed transitions such as
    # bedroom -> bathroom and picked-up-object -> later movement.
    all_terms = sorted({term for values in link_terms.values() for term in values})
    for term in all_terms:
        related = [source_id for source_id in anchored if term in link_terms[source_id]]
        if len(related) > 1:
            occurrences = [unit['id'] for unit in units
                           if term in set(re.findall(r'\w+', unit['text'].casefold()))]
            chosen = list(related[:max_chunks_per_cluster])
            neighbors = []
            positions = {source_id: index for index, source_id in enumerate(occurrences)}
            for source_id in related:
                position = positions.get(source_id)
                if position is None: continue
                for offset in (-1, 1):
                    other = position+offset
                    if 0 <= other < len(occurrences) and occurrences[other] not in chosen:
                        neighbors.append((abs(offset), source_order[occurrences[other]], occurrences[other]))
            for _, _, source_id in sorted(set(neighbors)):
                if len(chosen) >= max_chunks_per_cluster: break
                chosen.append(source_id)
            add_cluster(chosen, 'retrieval_entity_chain')

    # Singletons preserve strong local facts, but never consume the whole
    # cluster budget: cross-chunk entity chains are the fragmentation target.
    singleton_budget = min(len(ranked_anchors), max(2, max_clusters//4))
    for source_id in ranked_anchors[:singleton_budget]:
        if not add_cluster([source_id], 'retrieval_anchor_singleton'): break

    bridge_units = sorted((unit for unit in units if not unit['query_anchor']
                           and isinstance(unit.get('bridge_score'), (int, float))
                           and not isinstance(unit.get('bridge_score'), bool) and unit['bridge_score'] > 0),
                          key=lambda unit: (-unit['bridge_score'], source_order[unit['id']]))
    # Build minimal entity-linked anchor+bridge candidates. In addition to the
    # combined chain, keep each pair so one irrelevant anchor cannot poison an
    # otherwise sufficient Teacher context.
    for bridge in bridge_units:
        if len(clusters) >= max_clusters: break
        bridge_words = set(re.findall(r'\w+', bridge['text'].casefold()))
        related = [source_id for source_id in anchored if link_terms[source_id] & bridge_words]
        if not related: continue
        add_cluster(related[:max(1, max_chunks_per_cluster-1)]+[bridge['id']],
                    'retrieval_anchor_bridge_chain')
        nearest = min(related, key=lambda source_id: (
            abs(source_order[source_id]-source_order[bridge['id']]),
            source_order[source_id] > source_order[bridge['id']], source_order[source_id]))
        if len(clusters) < max_clusters:
            add_cluster([nearest, bridge['id']], 'retrieval_anchor_bridge')
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        ids = item.get('source_ids') if isinstance(item, dict) else None
        if (not isinstance(ids, list) or not ids
                or not all((isinstance(value, str) or type(value) is int) for value in ids)):
            rejected.append({'index': index, 'cluster': item, 'reason': 'invalid_source_id_list'}); continue
        add_cluster(ids, 'model_proposed', index=index)
    # Add unproposed high-ranked anchor pairs last; the cap prevents lexical
    # distractors from crowding out entity-linked bridge chains.
    for left, right in combinations(ranked_anchors[:singleton_budget], 2):
        if len(clusters) >= max_clusters: break
        add_cluster([left, right], 'retrieval_anchor_pair')
    for source_id in ranked_anchors[singleton_budget:]:
        if len(clusters) >= max_clusters: break
        add_cluster([source_id], 'retrieval_anchor_singleton')
    retrieval_seeded_count = sum(row['origin'].startswith('retrieval_') for row in clusters)
    call = ({**response, 'instruction': instruction, 'input': {'query': query, 'sources': units},
            'parsed': parsed, 'parse_status': parse_status,
            'max_completion_tokens': cluster_tokens, 'wall_seconds': time.monotonic()-started}
            if use_model_proposals else None)
    return {'clusters': clusters, 'audit': {'version': VERSION, 'query': query, 'call': call,
        'accepted_count': len(clusters), 'requested_max_clusters': max_clusters,
        'max_chunks_per_cluster': max_chunks_per_cluster, 'query_anchor_source_ids': anchored,
        'retrieval_seeded_count': retrieval_seeded_count,
        'model_accepted_count': sum(row['origin'] == 'model_proposed' for row in clusters),
        'rejections': rejected,
        'benchmark_answer_values_used': False, 'training_performed': False,
        **({} if use_model_proposals else {'model_proposals_enabled': False,
                                         'model_calls': 0, 'pair_synergy_model_judged': False}),
        'semantic_status': 'model_estimate' if use_model_proposals else 'heuristic_candidate',
        'limitations': [('Cluster usefulness and complementarity are frozen-model estimates.'
                         if use_model_proposals else
                         'Groups use deterministic source links, not a model judgment of joint sufficiency.'),
                        'Literal source IDs are validated, but cluster sufficiency is not guaranteed.']}}


async def frozen_text_call(engine, instruction, payload, *, max_tokens):
    """One ordinary frozen-model decode; no logit targets, student, or optimizer."""
    from .upstream_study import LocalStudyClient, _RUNTIMES
    key = str(id(engine)); _RUNTIMES[key] = engine
    client_type = getattr(engine, 'study_client_type', LocalStudyClient)
    client = client_type(client_type.Config(model_name=engine.tokenizer.name_or_path, runtime_key=key))
    try:
        response = await client.chat([[{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]],
            temperature=0., max_completion_tokens=max_tokens, top_logprobs=None)
        samples = response.samples
        result = {'response_count': len(samples), 'raw_output': None,
                  'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        if len(samples) != 1:
            result['unexpected_outputs'] = [sample.text for sample in samples]
            return result
        sample = samples[0]; ids = sample.token_ids
        n = len(ids) if ids is not None else 0
        eos = bool(n and ids[-1] in engine.eos_ids) if hasattr(engine, 'eos_ids') else None
        return {**result, 'raw_output': sample.text, 'eos_reached': eos,
                'token_budget_reached': n >= max_tokens, 'possibly_truncated': n >= max_tokens and eos is not True}
    finally:
        _RUNTIMES.pop(key, None)


async def build_answer_guided_dataset(engine, query, evidence_clusters, *, max_examples,
                                     answer_tokens=768, question_tokens=256, questions_per_answer=8,
                                     call_model=None, simple_two_stage=False):
    """Return JSON-serializable examples and audit from ordered ``id,text,source_chunks`` clusters.

    ``call_model`` is an optional async text transport with the same signature as
    frozen_text_call. It returns raw_output, response_count, and token usage.
    No benchmark answers are accepted. Extracted answers remain model estimates.
    The opt-in simple path performs answer extraction then question generation.
    Deterministic format checks remove non-informative answers, answer-type
    mismatches, source-deictic prompts, explicit assumptions, and direct target
    leakage. Quotes annotate source provenance; they do not establish semantic
    correctness or joint reasoning.
    """
    if not isinstance(query, str) or not query.strip(): raise ValueError('A nonempty real query is required')
    if type(simple_two_stage) is not bool: raise ValueError('simple_two_stage must be boolean')
    if any(type(n) is not int or n < 1 for n in
           (max_examples, answer_tokens, question_tokens, questions_per_answer)):
        raise ValueError('Example and generation budgets must be positive integers')
    clusters = json.loads(json.dumps(list(evidence_clusters), ensure_ascii=False, allow_nan=False))
    ids = []
    for cluster in clusters:
        if (not isinstance(cluster, dict) or not isinstance(cluster.get('id'), str) or not cluster['id'].strip()
                or not isinstance(cluster.get('text'), str) or not isinstance(cluster.get('source_chunks'), list)
                or not cluster['source_chunks']
                or any(not isinstance(chunk, dict) or not isinstance(chunk.get('text'), str)
                       or not chunk['text'] or chunk['text'] not in cluster['text']
                       or not isinstance(chunk.get('span'), list) or len(chunk['span']) != 2
                       or not all(type(value) is int for value in chunk['span'])
                       or not 0 <= chunk['span'][0] < chunk['span'][1]
                       or chunk['span'][1]-chunk['span'][0] != len(chunk['text'])
                       for chunk in cluster['source_chunks'])):
            raise ValueError('Each evidence cluster needs a stable string id, text, and source_chunks list')
        ids.append(cluster['id'])
    if len(set(ids)) != len(ids): raise ValueError('Evidence cluster IDs must be unique')
    call_model = call_model or frozen_text_call
    started = time.monotonic(); calls, rejected, examples, decisions, seen = [], [], [], [], {}

    async def invoke(stage, cluster, instruction, payload, budget):
        start = time.monotonic()
        response = await call_model(engine, instruction, payload, max_tokens=budget)
        if not isinstance(response, dict): response = {'response_count': 0, 'raw_output': None, 'transport_result_type': type(response).__name__}
        parsed, parse_status = (parse_object_with_status(response.get('raw_output'))
                                if response.get('response_count') == 1 else (None, 'unknown'))
        record = {**response, 'index': len(calls), 'stage': stage, 'evidence_id': cluster['id'],
                  'instruction': instruction, 'input': payload, 'max_completion_tokens': budget,
                  'wall_seconds': time.monotonic()-start, 'parsed': parsed,
                  'parse_status': parse_status}
        calls.append(record)
        return parsed, record['index']

    def reject(cluster, stage, reason, **extra):
        record = {'evidence_id': cluster['id'], 'stage': stage, 'reason': reason, **extra}
        rejected.append(record); decisions.append({**record, 'accepted': False})

    for cluster in clusters:
        if len(examples) == max_examples: break
        evidence = {name: cluster[name] for name in ('id', 'text', 'source_chunks')}
        for name in ('origin', 'query_complete_core_source_indices'):
            if name in cluster:
                evidence[name] = cluster[name]
        if not evidence['text'].strip():
            reject(cluster, 'input', 'empty_evidence'); continue
        extraction_view = focused_evidence_view(evidence, query, simple_two_stage=simple_two_stage)
        joint_instruction = (
            'Because this evidence has multiple ordered sources, do not merely repeat the answer from one source. '
            'Compare all sources and produce a joint conclusion that explicitly states how their query-relevant facts '
            'combine (for example, a sequence of state changes or a relational chain). The answer and quotes must use '
            'at least two different sources. If no such joint conclusion is supported, return skip=true. '
            if len(evidence['source_chunks']) > 1 else '')
        extraction_instruction = (
            'Extract one useful concrete answer or intermediate conclusion from this evidence for solving the real query. '
            'Return JSON only: {"skip":false,"answer":"a concrete answer or subconclusion",'
            '"quotes":["exact complete source sentence"],"why_useful":"brief explanation"}. '
            'If the evidence contains no useful supported information, return {"skip":true,"reason":"brief reason"}. '
            'Use only the supplied evidence, quote it exactly, and do not invent missing links. '
            'For single-source evidence, make answer the shortest standalone target value (for example a place, '
            'person, category, date, or number), rather than a sentence that repeats the query entity. '
            'In the quotes array, put raw source sentence text inside valid JSON strings; do not add an extra layer '
            'of quotation marks around a sentence. ' + joint_instruction +
            'When the evidence contains multiple sources, return a joint conclusion only if it genuinely uses facts '
            'from at least two sources, and quote at least one exact supporting sentence from each used source; '
            'otherwise return skip=true. '
            'The answer may be a prerequisite rather than the final answer to the real query. '
            'Return one object, not a checklist or multiple examples.')
        if simple_two_stage:
            extraction_instruction = (
                'Read the ordered evidence and extract one specific answer or intermediate conclusion useful '
                'for resolving the real query. Negative answers, uncertainty between stated alternatives, '
                'and facts that contradict the query proposition are useful information too: do not skip '
                'merely because the evidence does not confirm the proposition in the query. '
                'The extracted answer must be logically entailed by the supplied evidence, not merely '
                'compatible with it or possible under it. Preserve the exact epistemic strength and modality '
                'of the evidence. In particular, from "X is either A or B" never conclude that X is A or '
                'that X is B unless another supplied fact rules out the other alternative. If the real query '
                'asks about one unresolved alternative, extract that it is uncertain or only possible, rather '
                'than answering yes or no. If supplied sources conflict and the conflict cannot be resolved '
                'from their stated order or conditions, state the uncertainty or return skip=true. '
                'First try to extract the final answer to the real query from the complete ordered evidence. '
                'For state, location, possession, and inventory questions, treat the ordered source segments '
                'as an event sequence: a later explicit event supersedes an earlier state. Never collapse '
                'successive historical states into a simultaneous list or an "A or B" answer. Only when the '
                'final query cannot be resolved may you extract a necessary intermediate conclusion, and then '
                'state its time or condition so it cannot be mistaken for the final state. '
                'A necessary intermediate fact is useful even when it does not settle the final question. '
                'Combine source facts when needed; a useful fact from one source remains valid in a larger '
                'cluster and need not be forced into a joint conclusion. Preserve the order and stated '
                'conditions of events. Resolve references when the evidence permits; otherwise state what '
                'is known and what remains uncertain without inventing a missing link. '
                'Return one JSON object: {"skip":false,"answer":"the answer or intermediate conclusion",'
                '"quotes":["one exact source segment per string"],"why_useful":"brief explanation"}. '
                'Source segments separated by blank lines may have gaps in the original document; do not '
                'join them into one verbatim quote. Use raw segment text inside valid JSON strings. '
                'Only if the evidence offers no useful information for the query or its prerequisites, '
                'return {"skip":true,"reason":"brief reason"}.')
        answer, extraction_call = await invoke('extract_answer', cluster, extraction_instruction,
            {'query': query, 'evidence': extraction_view,
             'source_count': len(evidence['source_chunks'])}, answer_tokens)
        if answer is None:
            reject(cluster, 'extract_answer', 'missing_or_unparseable_response', call=extraction_call); continue
        if answer.get('skip') is True:
            reject(cluster, 'extract_answer', 'model_explicit_skip', call=extraction_call,
                   model_reason=answer.get('reason')); continue
        if answer.get('skip') is not False or not isinstance(answer.get('answer'), str) or not answer['answer'].strip():
            reject(cluster, 'extract_answer', 'invalid_or_empty_answer', call=extraction_call); continue
        if not simple_two_stage and re.search(r'\b(?:there|here|somewhere|location where|place where|that location|that place)\b',
                     answer['answer'], re.IGNORECASE):
            reject(cluster, 'extract_answer', 'unresolved_deictic_answer', call=extraction_call); continue
        quotes = literal_quotes(answer.get('quotes'), evidence['source_chunks'])
        if quotes is None and not simple_two_stage:
            reject(cluster, 'extract_answer', 'quote_not_in_this_evidence', call=extraction_call); continue
        quote_provenance_complete = quotes is not None
        if simple_two_stage and quotes is None:
            quotes = []
            reported = answer.get('quotes')
            for quote in reported if isinstance(reported, list) else []:
                matches = literal_quotes([quote], evidence['source_chunks'])
                if matches: quotes.extend(matches)
        covered_sources = {match['source_index'] for quote in quotes for match in quote['matches']}
        if not simple_two_stage and len(evidence['source_chunks']) > 1 and len(covered_sources) < 2:
            reject(cluster, 'extract_answer', 'multi_source_answer_not_joint', call=extraction_call,
                   covered_source_indices=sorted(covered_sources)); continue
        if not simple_two_stage and len(evidence['source_chunks']) > 1:
            query_content = {word for word in re.findall(r'\w+', query.casefold())
                             if word not in _QUERY_STOP and len(word) > 1}
            quote_links = []
            for quote in quotes:
                links = set()
                for raw in re.findall(r'[^\W_]+', quote['quote'], flags=re.UNICODE):
                    word = raw.casefold()
                    if raw[:1].isupper() or raw.isdigit() or word in query_content: links.add(word)
                quote_links.append(links)
            reached = {0}
            while True:
                expanded = reached | {j for i in reached for j, links in enumerate(quote_links)
                                      if quote_links[i].intersection(links)}
                if expanded == reached: break
                reached = expanded
            if len(reached) != len(quote_links):
                reject(cluster, 'extract_answer', 'multi_source_quotes_disconnected', call=extraction_call,
                       covered_source_indices=sorted(covered_sources)); continue
        answer_text = answer['answer'].strip()
        if simple_two_stage:
            reason = noninformative_answer(answer_text)
            if reason is not None:
                reject(cluster, 'extract_answer', reason, call=extraction_call)
                continue
        ordered_quotes = sorted(quotes, key=lambda quote: min(match['document_start'] for match in quote['matches']))
        requested_questions = min(questions_per_answer, max_examples-len(examples))
        candidate_questions = (min(8, max(requested_questions, 4*requested_questions))
                               if simple_two_stage else
                               min(16, max(requested_questions, 2*requested_questions)))
        question_instruction = (
            'Construct self-contained closed-book study questions whose answer is the supplied extracted answer. '
            f'Return JSON only: {{"questions":["question 1", ...]}} with exactly {candidate_questions} candidate questions. '
            'Use the real query to keep every question useful. Name necessary entities and conditions, but do not put '
            'the answer or all solution facts in a question. '
            'Use the same supplied evidence as answer extraction. The evidence and supporting quotes are in original '
            'document order: an earlier pickup followed by a later move must not become a pickup at the later destination. '
            'The extraction rationale is a proposal, not an additional source fact. '
            'Each question must preserve exactly the relation stated by the evidence and supporting quotes. '
            'Match the answer type exactly: a Where question needs a location answer; do not pair a place name '
            'with a request to identify a sentence. The supplied answer must directly answer the question as written. '
            'Use only entities, '
            'actions, conditions, and settings that occur in the real query or supporting quotes; lexical variety is '
            'unnecessary. Never invent a purpose, routine, job or work context, meeting, activity, observation, or '
            'reason for an event. Do not mention surrounding events, characters, or source labels. '
            'Do not add current, latest, final, recent, now, before, after, or another temporal/causal relation unless '
            'that relation is explicit in both the extracted answer and supporting quotes. A local intermediate fact '
            'must not be presented as the final answer to the real query. '
            'Do not refer to "the text above", copy source blocks, supply an answer, or ask for text reconstruction. '
            'The student will see only the question; it must recall the relevant evidence facts to answer. '
            'If the target cannot answer a supported question, return an empty questions list.')
        if simple_two_stage:
            question_instruction = (
                'Using the supplied evidence and extracted answer, construct self-contained study questions '
                'whose answer is exactly that extracted answer. Keep them useful for resolving the real query '
                'or learning its prerequisites. Name the necessary entities and conditions so that the Student '
                'can answer from remembered evidence without seeing the document. Preserve uncertainty, '
                'negation, event order and relational direction. When the answer combines several source facts, '
                'ask a question that requires that combination; do not force an unrelated source into it. '
                'Match the answer type exactly: if the extracted answer begins with Yes or No, every question '
                'must be a yes/no question, never Who/What/Where/When/Which/How. Do not mention the evidence, '
                'source, text, passage, statement, or "given information", and do not add an assumption. '
                'Do not put the target answer or all solution facts in the question. Do not invent facts, '
                'change the extracted answer, ask for source reconstruction, or refer to source labels. '
                f'Return JSON only: {{"questions":["question 1", ...]}} with {candidate_questions} candidate '
                'questions. If no meaningful question can be answered by the target from this evidence, '
                'return an empty questions list.')
        question, question_call = await invoke('construct_question', cluster, question_instruction,
            {'query': query, 'evidence_id': evidence['id'],
             'evidence': extraction_view,
             'supporting_quotes': [quote['quote'] for quote in ordered_quotes],
             'answer_reasoning': answer.get('why_useful'), 'extracted_answer': answer_text,
             'requested_question_candidates': candidate_questions,
             'maximum_accepted_questions': requested_questions}, question_tokens)
        values = question.get('questions') if isinstance(question, dict) else None
        # Accept the old singular key as a transport-format fallback, without
        # issuing a repair call or changing the requested protocol.
        if values is None and isinstance(question, dict) and isinstance(question.get('question'), str):
            values = [question['question']]
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            reject(cluster, 'construct_question', 'missing_or_empty_questions', call=question_call); continue
        accepted_here = []
        for variant_index, text in enumerate(values[:candidate_questions]):
            text = text.strip()
            reason = 'missing_or_empty_question' if not text else None
            if simple_two_stage and reason is None:
                reason = closed_book_question_rejection(
                    text, answer_text, [quote['quote'] for quote in ordered_quotes])
            elif not simple_two_stage:
                if reason is None: reason = query_request_rejection(text)
                if reason is None and '?' not in text and not _REQUEST_START.search(text): reason = 'not_a_question_or_request'
                if reason is None: reason = unsupported_question_claim(text, answer_text, query)
                if reason is None: reason = direct_answer_leakage(text, answer_text)
            normalized = ' '.join(text.casefold().split())
            if reason is None and normalized in seen: reason = 'duplicate_question'
            if reason:
                record = {'evidence_id': evidence['id'], 'stage': 'construct_question', 'reason': reason,
                          'call': question_call, 'variant_index': variant_index}
                if reason == 'duplicate_question': record['previous_evidence_id'] = seen[normalized]
                rejected.append(record); continue
            seen[normalized] = evidence['id']
            examples.append({'question': text, 'answer': answer_text, 'evidence': evidence,
                             'answer_source_quotes': quotes, 'why_useful': answer.get('why_useful'),
                             'semantic_status': 'model_estimate', 'source_facts_verified': False,
                             'call_indices': [extraction_call, question_call]})
            if simple_two_stage:
                examples[-1].update(simple_two_stage=True,
                    method_version='answer-guided-simple-two-stage-v3',
                    raw_answer_source_quotes=answer.get('quotes'),
                    quote_provenance_complete=quote_provenance_complete,
                    source_quote_matching_used_for_admission=False,
                    joint_source_support_verified=False)
            accepted_here.append(len(examples)-1)
            if len(examples) == max_examples or len(accepted_here) == requested_questions: break
        if not accepted_here:
            decisions.append({'evidence_id': evidence['id'], 'stage': 'construct_question',
                              'reason': 'no_valid_question_variants', 'call': question_call, 'accepted': False})
        else:
            decisions.append({'evidence_id': evidence['id'], 'accepted': True,
                              'example_indices': accepted_here})
    known_cost = lambda field: sum(row[field] for row in calls if type(row.get(field)) is int and row[field] >= 0)
    unknown_cost = lambda field: sum(type(row.get(field)) is not int or row[field] < 0 for row in calls)
    audit = {'version': VERSION, 'query': query, 'requested_max_examples': max_examples,
        'questions_per_answer': questions_per_answer, 'accepted_count': len(examples),
        'shortfall': max_examples-len(examples), 'candidate_evidence_ids': ids, 'decisions': decisions,
        'unprocessed_evidence_ids': ids[len(decisions):], 'calls': calls, 'rejections': rejected,
        'call_count': len(calls), 'maximum_calls': 2*len(clusters), 'wall_seconds': time.monotonic()-started,
        'cost': {field: {'known_sum': known_cost(field), 'unknown_calls': unknown_cost(field),
                        'total': None if unknown_cost(field) else known_cost(field)}
                 for field in ('prompt_tokens', 'completion_tokens')},
        'benchmark_answer_values_used': False, 'teacher_forced_targets_created': False,
        'training_performed': False, 'semantic_status': 'model_estimate',
        'limitations': ['Literal quotes establish only occurrence in the supplied cluster, not semantic correctness.',
            'Source chunk provenance is preserved as supplied by the caller; no hidden benchmark answers are consulted.',
            'Question-answer consistency and usefulness are model estimates without an independent checker.',
            'Direct answer leakage uses a conservative lexical guard, not a complete semantic test.',
            'Invalid samples are skipped; no repair loop or target-count guarantee is applied.']}
    if simple_two_stage:
        audit.update(version='answer-guided-simple-two-stage-v3', simple_two_stage=True,
            semantic_heuristic_admission=False, source_quote_matching_used_for_admission=False,
            joint_source_support_verified=False, additional_verification_model_calls=0,
            limitations=[
                'Answers and questions are model estimates, not independently verified facts.',
                'Source quote matches are provenance annotations only; unmatched raw quotes are retained.',
                'A multi-source input does not establish that its answer requires joint reasoning.',
                ('Only deterministic interface checks and within-case deduplication filter outputs; '
                 'no semantic verifier or extra model call is used.'),
                'Model skips or malformed responses can still leave a shortfall; no repair calls are made.'])
    return {'examples': examples, 'audit': audit}
