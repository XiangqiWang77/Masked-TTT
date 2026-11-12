"""Local Qwen3-0.6B reranking of canonical source chunks, without answer labels.

The older source-score files supply text offsets and case identity only. Their
model decisions and relevance values never participate in this ranking.
"""
import gc
import json
import math
from pathlib import Path
import re
import time

from experiments import build_synthetic_comparison as B


REPO = 'Qwen/Qwen3-Reranker-0.6B'
REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based '
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PAIR_INSTRUCTION = ('Given a question and two source passages, determine whether BOTH passages jointly '
    'provide complementary facts, prerequisites, or a bridging relationship useful for answering the '
    'question. Each passage must contribute; one relevant passage paired with an irrelevant or '
    'merely redundant passage does not meet this requirement.')
VERSION = 'qwen-reranker-canonical-chunks-v1'


def ranking_prompt(query, passage, instruction=INSTRUCTION):
    if not query.strip() or not passage.strip() or not instruction.strip():
        raise ValueError('Reranking requires a query, a whole passage and an instruction')
    return PREFIX+f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {passage}'+SUFFIX


def canonical_chunks(case, plan, mapping):
    path = Path(plan['shared_scores'])/'scores'/mapping[case['key']]
    old = B.read(path)
    if (old.get('schema') != 'masked-ttt-source-scores-v1' or old.get('complete') is not True
            or old.get('case_key') != case['key'] or old.get('question') != case['query']
            or old.get('document_sha256') != B.text_hash(case['document'])
            or old.get('reference_answers_used') is not False):
        raise ValueError('Canonical source identity does not match the label-free case')
    spans = old.get('source_spans')
    if not isinstance(spans, list) or not spans:
        raise ValueError('Canonical source spans are missing')
    previous = -1
    for span in spans:
        if (not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span)
                or not 0 <= span[0] < span[1] <= len(case['document']) or span[0] <= previous):
            raise ValueError('Invalid canonical source offsets')
        previous = span[0]
    return spans, [case['document'][a:b] for a, b in spans], path


def model_contract(retrieval):
    if retrieval.get('repo') != REPO or retrieval.get('revision') != REVISION:
        raise ValueError('Require the registered official Qwen3-Reranker-0.6B revision')
    path = Path(retrieval['model']).resolve(); expected = retrieval['model_assets_sha256']
    if not expected or not any(name.endswith('.safetensors') for name in expected):
        raise ValueError('Reranker weight hashes are required')
    for name, checksum in expected.items():
        if Path(name).name != name or B.digest(path/name) != checksum:
            raise ValueError('Reranker model artifact changed: '+name)
    config = B.read(path/'config.json')
    if config.get('model_type') != 'qwen3':
        raise ValueError('The registered reranker must be Qwen3')
    return {'repo': REPO, 'revision': REVISION, 'model': str(path), 'model_assets_sha256': expected,
        'instruction': retrieval.get('instruction', INSTRUCTION),
        'pair_instruction': retrieval.get('pair_instruction', PAIR_INSTRUCTION), 'prompt_format': VERSION,
        'prefix': PREFIX, 'suffix': SUFFIX, 'score': 'softmax([logit_no,logit_yes])[yes]',
        'max_length': retrieval.get('max_length', 4096), 'dtype': 'bfloat16'}


def score_contract(case, spans, canonical_path, model):
    return {'case': case['key'], 'query_sha256': B.text_hash(case['query']),
        'document_sha256': B.text_hash(case['document']), 'source_spans': spans,
        'canonical_span_file': str(canonical_path), 'canonical_span_file_sha256': B.digest(canonical_path),
        'model': model, 'legacy_relevance_values_used': False, 'benchmark_answers_used': False}


def validate_scores(record, contract):
    if (record.get('schema') != VERSION or record.get('contract') != contract
            or record.get('identity') != B.fingerprint(contract) or record.get('complete') is not True):
        raise ValueError('Cached reranker scores belong to another model, query or document')
    rows = record.get('records', [])
    if len(rows) != len(contract['source_spans']):
        raise ValueError('Incomplete reranker scores')
    for index, row in enumerate(rows):
        score = row.get('score')
        if (row.get('source_index') != index or row.get('span') != contract['source_spans'][index]
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
                or any(not isinstance(row.get(k), (int, float)) or isinstance(row.get(k), bool)
                       or not math.isfinite(row[k]) for k in ('logit_no', 'logit_yes'))):
            raise ValueError('Invalid reranker score or source position')
        delta = row['logit_yes']-row['logit_no']
        expected = 1/(1+math.exp(-max(-700, min(700, delta))))
        if abs(score-expected) > 2e-6:
            raise ValueError('Saved reranker score does not match its yes/no logits')
    return rows


def prepare_reranker_scores(manifest, planned_cases, output, tokenizer=None, stop_requested=None):
    """Score each rank's cases, then release the small model before loading vLLM."""
    from contextualized.data import read_indexed_case
    retrieval = manifest['retrieval']; model_info = model_contract(retrieval)
    batch_size = retrieval.get('batch_size', 64); max_length = model_info['max_length']
    if type(batch_size) is not int or batch_size < 1 or type(max_length) is not int or max_length < 1:
        raise ValueError('Reranker batch and context budgets must be positive integers')
    plan = B.read(manifest['full_plan']); index = {r['key']: r for r in B.read(plan['source_index'])}
    mapping = B.read(Path(plan['shared_scores'])/'case_to_score_file.json')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    paths, missing = {}, []
    for item in planned_cases:
        case = read_indexed_case(plan, index, item['key'])
        spans, _, canonical = canonical_chunks(case, plan, mapping)
        contract = score_contract(case, spans, canonical, model_info)
        path = output/(case['key']+'.json'); paths[case['key']] = str(path)
        if path.exists(): validate_scores(B.read(path), contract)
        else: missing.append(item)
    if not missing: return paths
    if stop_requested is not None and stop_requested():
        return {key: path for key, path in paths.items() if Path(path).exists()}
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device('cuda')
    reranker_tokenizer = AutoTokenizer.from_pretrained(model_info['model'], padding_side='left',
        local_files_only=True, trust_remote_code=False)
    if reranker_tokenizer.pad_token_id is None:
        reranker_tokenizer.pad_token_id = reranker_tokenizer.eos_token_id
    no, yes = [reranker_tokenizer.convert_tokens_to_ids(word) for word in ('no', 'yes')]
    if no == yes or any(reranker_tokenizer.encode(word, add_special_tokens=False) != [token]
                        for word, token in [('no', no), ('yes', yes)]):
        raise ValueError('Reranker labels must be distinct single yes/no tokens')
    model = None; encoded_batch = output_state = logits = head = bias = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_info['model'], torch_dtype=torch.bfloat16,
            device_map={'': device}, attn_implementation='flash_attention_2',
            local_files_only=True, trust_remote_code=False).eval().requires_grad_(False)
        head = model.lm_head.weight[[no, yes]]
        bias = model.lm_head.bias
        if bias is not None: bias = bias[[no, yes]]
        for item in missing:
            if stop_requested is not None and stop_requested(): break
            case = read_indexed_case(plan, index, item['key'])
            spans, passages, canonical = canonical_chunks(case, plan, mapping)
            contract = score_contract(case, spans, canonical, model_info)
            started = time.monotonic(); records = []; input_tokens = 0
            for start in range(0, len(passages), batch_size):
                texts = passages[start:start+batch_size]
                tokens = [reranker_tokenizer.encode(ranking_prompt(case['query'], text, model_info['instruction']),
                                                    add_special_tokens=False) for text in texts]
                if any(len(row) > min(max_length, model.config.max_position_embeddings) for row in tokens):
                    raise ValueError('Reranker input exceeds its registered limit; no source truncation permitted')
                input_tokens += sum(map(len, tokens))
                with torch.inference_mode():
                    encoded_batch = reranker_tokenizer.pad({'input_ids': tokens}, padding=True,
                                                          return_tensors='pt').to(device)
                    positions = (encoded_batch.attention_mask.cumsum(-1)-1).clamp_min(0)
                    output_state = model.model(**encoded_batch, position_ids=positions, use_cache=False)
                    logits = F.linear(output_state.last_hidden_state[:, -1], head, bias).float()
                    values = logits.cpu().tolist(); scores = logits.softmax(-1)[:, 1].cpu().tolist()
                for offset, ((logit_no, logit_yes), score) in enumerate(zip(values, scores)):
                    i = start+offset
                    records.append({'source_index': i, 'span': spans[i], 'logit_no': logit_no,
                        'logit_yes': logit_yes, 'score': score, 'input_tokens': len(tokens[offset])})
                encoded_batch = output_state = logits = positions = None
            record = {'schema': VERSION, 'contract': contract, 'identity': B.fingerprint(contract),
                'complete': True, 'records': records, 'wall_seconds': time.monotonic()-started,
                'input_tokens': input_tokens, 'generated_tokens': 0, 'batch_size': batch_size,
                'score_is_calibrated_evidence_correctness': False}
            validate_scores(record, contract); B.write(paths[case['key']], record)
            print(json.dumps({'event': 'qwen_reranker_case_ready', 'case': case['key'],
                'chunks': len(records), 'seconds': record['wall_seconds']}), flush=True)
    finally:
        model = encoded_batch = output_state = logits = head = bias = None
        gc.collect(); torch.cuda.empty_cache()
    return {key: path for key, path in paths.items() if Path(path).exists()}


def auxiliary_features(query, passages):
    """Keep lexical/entity scores honestly named; they do not replace reranking."""
    from contextualized.data import STOP_WORDS, BRIDGE_STOP_WORDS
    words = [re.findall(r'\w+', text.casefold()) for text in passages]
    query_terms = {w for w in re.findall(r'\w+', query.casefold()) if w not in STOP_WORDS and len(w) > 1}
    frequency = {t: sum(t in part for part in words) for t in query_terms}
    lexical = [sum((1+math.log(part.count(t)))*math.log(1+(len(words)+1)/(frequency[t]+1))
                   for t in sorted(query_terms) if t in part)/math.sqrt(max(1, len(part))) for part in words]
    bridge = set()
    for text in passages:
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
            if not query_terms.intersection(re.findall(r'\w+', sentence.casefold())): continue
            for raw in re.findall(r'[^\W_]+', sentence):
                word = raw.casefold()
                if (word not in query_terms and word not in BRIDGE_STOP_WORDS
                        and (raw[:1].isupper() or raw.isdigit())):
                    bridge.add(word)
    bridge_frequency = {t: sum(t in part for part in words) for t in bridge}
    bridge = set(sorted(bridge, key=lambda t: (bridge_frequency[t], t))[:32])
    bridge_score = [sum(math.log(1+(len(words)+1)/(bridge_frequency[t]+1))
                        for t in sorted(bridge) if t in part)/math.sqrt(max(1, len(part))) for part in words]
    return lexical, bridge_score, sorted(query_terms), sorted(bridge)


def select_support(passages, scores, bridge_scores, tokenizer, *, candidates, support_tokens):
    if candidates < 1 or support_tokens < 1 or not passages or len(passages) != len(scores):
        raise ValueError('Invalid reranker candidate inputs')
    rank = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    seeds = rank[:candidates]
    neighbors = list(dict.fromkeys(j for i in seeds for j in (i-1, i+1) if 0 <= j < len(passages)))
    bridges = [i for i in rank if bridge_scores[i] > 0]
    priority = list(dict.fromkeys(seeds+bridges+neighbors+rank))
    selected = []; tokens = 0
    for i in priority:
        trial = sorted(selected+[i]); text = '\n\n'.join(passages[j] for j in trial)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n <= support_tokens:
            selected = trial; tokens = n
    if not selected or not set(seeds).issubset(selected):
        raise ValueError('Teacher support budget cannot hold the reranker seed pool')
    return selected, seeds, tokens


def build_reranked_contexts(case, tokenizer, args, plan, mapping, score_path):
    spans, passages, canonical = canonical_chunks(case, plan, mapping)
    scored = B.read(score_path); model = scored.get('contract', {}).get('model', {})
    if model.get('repo') != REPO or model.get('revision') != REVISION:
        raise ValueError('Evidence scores are not from the requested Qwen reranker')
    contract = score_contract(case, spans, canonical, model); rows = validate_scores(scored, contract)
    scores = [row['score'] for row in rows]
    lexical, bridges, query_terms, bridge_terms = auxiliary_features(case['query'], passages)
    chosen, seeds, tokens = select_support(passages, scores, bridges, tokenizer,
        candidates=args.candidates, support_tokens=args.support_tokens)
    rank = {i: r for r, i in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))}
    encoded = tokenizer(case['document'], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if ids != tokenizer.encode(case['document'], add_special_tokens=False):
        raise ValueError('Original chunk tokenizer offsets differ')
    original = []
    for start in range(0, len(ids), args.chunk_tokens):
        end = min(start+args.chunk_tokens, len(ids)); char_span = [offsets[start][0], offsets[end-1][1]]
        original.append({'context_id': f'original_chunk_{len(original):04d}', 'kind': 'decoded_token_chunk',
            'text': tokenizer.decode(ids[start:end]), 'document_token_span': [start, end],
            'document_char_span': char_span, 'source_chunks': [{'source_index': i, 'span': span}
                for i, span in enumerate(spans) if span[0] < char_span[1] and span[1] > char_span[0]]})
    broad = {'context_id': 'broad', 'kind': 'qwen_reranked_original_spans',
        'text': '\n\n'.join(passages[i] for i in chosen),
        'source_chunks': [{'source_index': i, 'span': spans[i]} for i in chosen]}
    retrieval = {'method': 'qwen3_reranker_rank_plus_entity_bridges_and_neighbors',
        'model_relevance_scores_used': True, 'legacy_relevance_values_used': False,
        'source_chunks': len(spans), 'query_terms': query_terms, 'bridge_terms': bridge_terms,
        'candidate_source_indices': chosen, 'seed_candidate_source_indices': seeds,
        'support_pool_used_as_candidates': True, 'support_source_indices': chosen,
        'support_tokens': tokens, 'support_token_limit': args.support_tokens,
        'canonical_span_file': str(canonical), 'reranker_score_file': str(Path(score_path).resolve()),
        'reranker_score_file_sha256': B.digest(score_path), 'reranker': model,
        'reference_answers_used': False, 'recall_guaranteed': False,
        'cluster_construction_note': 'Qwen ranks the candidate pool; lexical/entity auxiliary features remain available to the unchanged cluster constructor.'}
    catalog_chunks = [{'source_index': i, 'span': spans[i], 'text': passages[i],
        'lexical_score': lexical[i], 'bridge_score': bridges[i], 'reranker_score': scores[i],
        'seed_rank': rank[i]} for i in range(len(spans))]
    return original, [broad], {'retrieval': retrieval, 'canonical_span_file_sha256': B.digest(canonical),
        'source_chunks': catalog_chunks, 'contexts': original+[broad], 'source_facts_verified': False,
        'source_trace_means': 'Original text locations, not evidence sufficiency or truth.'}
