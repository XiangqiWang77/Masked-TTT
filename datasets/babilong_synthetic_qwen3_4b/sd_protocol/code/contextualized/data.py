"""Load the existing label-free BabILong cases and frozen selection scores."""
from hashlib import sha256
import json
import math
import re
from pathlib import Path


REFERENCE_KEYS = frozenset({"answer", "answers", "reference", "references", "gold", "target", "label"})
STOP_WORDS = frozenset('a an the is are was were be been to of in on at for from by and or what where who which when how does do did has have had with before after'.split())
BRIDGE_STOP_WORDS = STOP_WORDS | frozenset('''
    about also any around as back because belong belonged belonging belongs but can could
    down each first get got into journeyed later left made make moved moving near not now
    other picked put said same since some then there these they this through took travelled
    travel up went while would your their him her his she its them was were will here
'''.split())


def read_indexed_case(plan, index, key):
    item = index[key]
    with open(plan["source_cases"], "rb") as stream:
        stream.seek(item["offset"])
        raw = stream.read(item["bytes"])
    case = json.loads(raw)
    if (len(raw) != item["bytes"] or case.get("key") != key or
            REFERENCE_KEYS.intersection(case) or
            case["prompt"].count(case["document"]) != 1 or
            case["query"] not in case["prompt"]):
        raise ValueError(f"Invalid or non-label-free indexed case: {key}")
    return case


def select_document(case, score_dir, mapping, tokenizer, fraction, max_tokens):
    if not 0 < fraction <= 1 or max_tokens < 1:
        raise ValueError("Invalid selection fraction or token budget")
    path = Path(score_dir) / "scores" / mapping[case["key"]]
    data = json.loads(path.read_text())
    digest = sha256(case["document"].encode()).hexdigest()
    if (data.get("schema") != "masked-ttt-source-scores-v1" or
            data.get("complete") is not True or data.get("case_key") != case["key"] or
            data.get("reference_answers_used") is not False or
            data.get("question") != case["query"] or data.get("document_sha256") != digest):
        raise ValueError("Cached source scores do not match this case")
    spans, records = data["source_spans"], data["records"]
    if len(spans) != len(records) or not records:
        raise ValueError("Incomplete cached source scores")
    ranked = []
    previous = -1
    for index, ((start, end), record) in enumerate(zip(spans, records)):
        prob = record.get("decision_probabilities", {}).get("yes")
        if (record.get("chunk") != index or record.get("utility_output_valid") is not True or
                not isinstance(prob, (float, int)) or not math.isfinite(prob) or
                not 0 <= prob <= 1 or not 0 <= start < end <= len(case["document"]) or
                start <= previous):
            raise ValueError("Invalid cached score or source span")
        previous = start
        ranked.append((index, float(prob)))
    wanted = max(1, math.ceil(fraction * len(ranked)))
    chosen = []
    used = 0
    for index, _ in sorted(ranked, key=lambda x: (-x[1], x[0]))[:wanted]:
        start, end = spans[index]
        text = case["document"][start:end]
        size = len(tokenizer.encode(text, add_special_tokens=False))
        if size and used + size <= max_tokens:
            chosen.append(index)
            used += size
    if not chosen:
        raise ValueError("No selected source chunk fits the teacher budget")
    # Separators and boundary tokenization also consume space. Drop the lowest
    # ranked retained chunks until the actual concatenation fits the budget.
    while chosen:
        ordered = sorted(chosen)
        text = "\n\n".join(case["document"][spans[i][0]:spans[i][1]] for i in ordered)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual <= max_tokens:
            break
        chosen.pop()
    if not chosen:
        raise ValueError("No selected source chunk fits after tokenizer-boundary accounting")
    used = sum(len(tokenizer.encode(case["document"][spans[i][0]:spans[i][1]],
                                    add_special_tokens=False)) for i in ordered)
    return text, {"source_chunks": len(ranked), "selected_chunk_indices": ordered,
                  "selected_spans": [spans[i] for i in ordered],
                  "selected_token_count_without_separators": used,
                  "selected_token_count": actual,
                  "selection_fraction": fraction, "teacher_max_source_tokens": max_tokens,
                  "cached_scoring_seconds": data.get("scoring_seconds"),
                  "cached_estimated_cost_usd": data.get("estimated_cost_usd"),
                  "precomputed_score_file": str(path), "reference_answers_used": False}


def relay_groups(document, spans, tokenizer, max_tokens):
    """Pack selected chunks in source order without introducing new evidence."""
    if max_tokens < 1 or not spans:
        raise ValueError("Relay requires selected spans and a positive token budget")
    groups, current = [], []
    for start, end in spans:
        text = document[start:end]
        size = len(tokenizer.encode(text, add_special_tokens=False))
        if not size or size > max_tokens:
            raise ValueError("A selected chunk is empty or exceeds the relay budget")
        candidate = "\n\n".join(current + [text])
        if current and len(tokenizer.encode(candidate, add_special_tokens=False)) > max_tokens:
            groups.append("\n\n".join(current))
            current = []
        current.append(text)
    if current:
        groups.append("\n\n".join(current))
    return groups


def lexical_candidate_pool(case, score_dir, mapping, tokenizer, *, candidate_count, support_tokens,
                           retain_support_pool=False):
    """Retrieve a small deterministic candidate pool without LLM relevance decisions.

    Reuse only the prior canonical source spans and identity hashes. Rank by
    query-term TF-IDF, include neighboring chunks for cross-chunk links, and
    keep a broader source view for producing independently supported probes.
    ``retain_support_pool`` makes that already retrieved view the Teacher
    candidate pool, avoiding a second small top-k cutoff that discards bridges.
    """
    if candidate_count < 1 or support_tokens < 1:
        raise ValueError('Candidate count and support budget must be positive')
    path = Path(score_dir) / 'scores' / mapping[case['key']]
    data = json.loads(path.read_text())
    digest = sha256(case['document'].encode()).hexdigest()
    if (data.get('schema') != 'masked-ttt-source-scores-v1' or data.get('complete') is not True or
            data.get('case_key') != case['key'] or data.get('question') != case['query'] or
            data.get('document_sha256') != digest or data.get('reference_answers_used') is not False):
        raise ValueError('Canonical source spans do not match this case')
    spans = data['source_spans']
    if not spans:
        raise ValueError('Source contains no canonical chunks')
    previous = -1
    chunks = []
    for start,end in spans:
        if not 0 <= start < end <= len(case['document']) or start <= previous:
            raise ValueError('Invalid canonical source span')
        previous = start
        chunks.append(case['document'][start:end])
    tokenize_words = lambda text: re.findall(r"[\w]+", text.casefold())
    terms = sorted({w for w in tokenize_words(case['query']) if w not in STOP_WORDS and len(w) > 1})
    words = [tokenize_words(text) for text in chunks]
    doc_frequency = {term:sum(term in part for part in words) for term in terms}
    scores = []
    for part in words:
        freq = {term:part.count(term) for term in terms}
        value = sum((1 + math.log(count)) * math.log(1 + (len(words)+1)/(doc_frequency[term]+1))
                    for term,count in freq.items() if count)
        scores.append(value / math.sqrt(max(1, len(part))))
    rank = sorted(range(len(chunks)), key=lambda i:(-scores[i], i))
    positive = [i for i in rank if scores[i] > 0]
    # A query-mentioned entity can introduce another entity in the same
    # source sentence (e.g. a person picked up an object). Search for that
    # bridge throughout the document before selecting the compact pool.
    bridge = set()
    for chunk in chunks:
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', chunk):
            if not set(tokenize_words(sentence)).intersection(terms):
                continue
            for raw in re.findall(r'[^\W_]+', sentence, flags=re.UNICODE):
                word = raw.casefold()
                if word in terms or word in BRIDGE_STOP_WORDS:
                    continue
                if len(word) > 2 or (len(word) == 1 and raw.isupper() and raw not in {'A','I'}):
                    bridge.add(word)
    bridge_frequency = {term:sum(term in part for part in words) for term in bridge}
    bridge = set(sorted(bridge,key=lambda x:(bridge_frequency[x],x))[:32])
    bridge_scores = []
    for part in words:
        score = sum(math.log(1 + (len(words)+1)/(bridge_frequency[term]+1))
                    for term in bridge if term in part)
        bridge_scores.append(score / math.sqrt(max(1,len(part))))
    bridge_rank = sorted(range(len(chunks)),key=lambda i:(-bridge_scores[i],i))
    seeds = positive[:max(1, (candidate_count+2)//3)]
    if not seeds:
        seeds = [len(chunks)//2]
    chosen = list(seeds)
    for i in bridge_rank:
        if len(chosen) >= min(candidate_count, len(seeds)+max(1,candidate_count//3)):
            break
        if bridge_scores[i] > 0 and i not in chosen:
            chosen.append(i)
    bridge_seeds = [i for i in chosen if i not in seeds]
    for center in seeds + bridge_seeds:
        for neighbor in (center-1, center+1):
            if 0 <= neighbor < len(chunks) and neighbor not in chosen and len(chosen) < candidate_count:
                chosen.append(neighbor)
    for i in rank:
        if len(chosen) >= min(candidate_count, len(chunks)):
            break
        if i not in chosen:
            chosen.append(i)
    chosen.sort()
    candidates = [{'source_index':i, 'span':spans[i], 'text':chunks[i], 'lexical_score':scores[i],
                   'bridge_score':bridge_scores[i]}
                  for i in chosen]

    # Build a broader, chronologically ordered support view, including the
    # candidate pool first. It remains independent of each candidate teacher.
    support_rank = chosen + [i for i in bridge_rank if bridge_scores[i] > 0 and i not in chosen]
    support_rank += [i for i in rank if i not in support_rank]
    selected, tokens = [], 0
    for i in support_rank:
        text = chunks[i]
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if not n or n > support_tokens or tokens + n > support_tokens:
            continue
        trial = sorted(selected + [i])
        trial_text = '\n\n'.join(chunks[j] for j in trial)
        actual = len(tokenizer.encode(trial_text, add_special_tokens=False))
        if actual <= support_tokens:
            selected = trial
            tokens = actual
    if not selected or any(c['source_index'] not in selected for c in candidates):
        raise ValueError('The support budget cannot hold the full candidate pool')
    support_groups = [chunks[i] for i in selected]
    seed_pool = chosen
    if retain_support_pool:
        chosen = selected
        candidates = [{'source_index':i, 'span':spans[i], 'text':chunks[i], 'lexical_score':scores[i],
                       'bridge_score':bridge_scores[i]} for i in chosen]
    return candidates, support_groups, {
        'method':'lexical_tfidf_plus_source_bridges_and_adjacent_chunks', 'model_relevance_scores_used':False,
        'source_chunks':len(chunks), 'query_terms':terms, 'candidate_source_indices':chosen,
        'seed_candidate_source_indices':seed_pool, 'support_pool_used_as_candidates':retain_support_pool,
        'bridge_terms':sorted(bridge),'bridge_seed_indices':bridge_seeds,
        'support_source_indices':selected, 'support_tokens':tokens, 'support_token_limit':support_tokens,
        'canonical_span_file':str(path), 'reference_answers_used':False,
    }


def task_without_document(case, question=None):
    body = case["prompt"].replace(case["document"], "[The context is provided above.]", 1)
    if question is not None:
        start = body.rfind(case["query"])
        if start < 0:
            raise ValueError("Question is absent from the label-free task prompt")
        body = body[:start] + question + body[start + len(case["query"]):]
    return body + case.get("answer_prefix", "")
