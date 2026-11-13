"""Thirty complete practice trajectories per request, without query fallback.

The query-blind arm calls the released SelfStudySynthesizer. The specific arm
retrieves source groups, extracts useful conclusions, then constructs questions.
Both cache real frozen-Teacher top-20 probabilities for the same native KV SD.
"""
import asyncio
import hashlib
import json
import re
from pathlib import Path
import time

from experiments.run_figure4_contextual import write
from experiments.retrain_plan import sha_file

VERSION = 'answer-first-30-unabridged-v1'


def key(text):
    return ' '.join(text.casefold().split())


def public_case(case):
    """Deliberately omit reference, options' answer key, and all metric fields."""
    return {name: case[name] for name in ('id', 'document', 'query', 'prompt')}


def paths(config, row):
    root = Path(config['output']) / 'synthetic' / row['dataset'] / row['case']
    return root, {f'{arm}_{suffix}': str(root / f'{arm}.{extension}')
        for arm in ('specific', 'agnostic')
        for suffix, extension in (('json', 'json'), ('targets', 'targets.pkl'))}


def save_targets(rows, path):
    from baselines.cartridge_data import _save_targets
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_targets(rows, path)


def finished(sample, eos_ids):
    return bool(sample.token_ids and sample.token_ids[-1] in eos_ids and sample.text.strip())


def valid_conversation(row, eos_ids):
    answer = row.messages[-1]
    return bool(answer.token_ids and answer.token_ids[-1] in eos_ids
                and answer.content.strip() and answer.top_logprobs is not None)


def validate_pair_manifest(row, manifest):
    minimum = row.get('synthetic_questions', 30)
    actual = manifest.get('actual_counts', {arm:manifest.get('count_per_arm', 0)
        for arm in ('specific','agnostic')})
    if manifest.get('complete') is not True or any(actual.get(arm,0) < minimum for arm in ('specific','agnostic')):
        raise ValueError('Both synthetic arms must meet the minimum complete-trajectory count')
    for name in ('case', 'dataset', 'document_sha256', 'prompt_sha256'):
        if row[name] != manifest[name]:
            raise ValueError('Synthetic pair identity mismatch')
    for name, path in manifest['data_files'].items():
        if sha_file(path) != manifest['data_sha256'][name]:
            raise ValueError('Synthetic artifact changed after completion')


def prepare_sources(config, rows, cases, deadline):
    from transformers import AutoTokenizer
    from experiments.question_guided_candidates import FrozenReranker, craft_evidence
    from experiments.qwen_reranker_candidates import INSTRUCTION, PAIR_INSTRUCTION
    from experiments.run_figure4_contextual import chunk_document, lexical_candidates
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    info = dict(model=config['reranker_model'], instruction=INSTRUCTION,
                pair_instruction=PAIR_INSTRUCTION, max_length=4096)
    with FrozenReranker(info, batch_size=config.get('reranker_batch_size', 4)) as reranker:
        chunk_cache = {}
        for row in rows:
            if time.time() >= deadline:
                raise TimeoutError('Source preparation deadline')
            root, _ = paths(config, row)
            source_path = root / 'sources.json'
            if source_path.exists():
                previous = json.loads(source_path.read_text())
                if previous['document_sha256'] != row['document_sha256'] or previous['prompt_sha256'] != row['prompt_sha256']:
                    raise ValueError('Source cache identity changed')
                continue
            case = public_case(cases[row['dataset']][row['case']])
            if row['document_key'] not in chunk_cache:
                chunk_cache[row['document_key']] = chunk_document(tokenizer, case['document'], 640)
            chunks = chunk_cache[row['document_key']]
            candidates = lexical_candidates(case['prompt'], chunks, maximum=64)
            retrieved = craft_evidence(case['prompt'], [c['text'] for c in candidates],
                [c['span'] for c in candidates], reranker.score,
                dict(candidate_pool=16, pair_candidates=32, max_evidence=32),
                score_pairs=lambda q, p: reranker.score(q, p, instruction=PAIR_INSTRUCTION))
            source_record = dict(version=VERSION, document_sha256=row['document_sha256'],
                prompt_sha256=row['prompt_sha256'], query=case['prompt'],
                evidence=retrieved['selected_evidence'], retrieval=retrieved,
                document_chunks=len(chunks), candidate_chunks=len(candidates),
                source_text_abridged=False, benchmark_answer_used=False)
            if config.get('reuse_existing_questions'):
                from experiments.reuse_synthesis_questions import include_reused_evidence
                source_record = include_reused_evidence(root, source_record)
            write(source_path, source_record)
            print(json.dumps(dict(event='sources_ready', dataset=row['dataset'],
                case=row['case'], groups=len(retrieved['selected_evidence']))), flush=True)


ANSWER_INSTRUCTION = '''Read the evidence and extract up to three DISTINCT concrete conclusions
or intermediate facts that help resolve the real query. Preserve negation,
conditions, temporal order and uncertainty. For a translation request, study
the requested sentence's words and constructions, rather than unrelated grammar.
For multiple sources, prefer a useful conclusion that combines their information
when supported; never invent a missing link. Return JSON only:
{"answers":[{"answer":"conclusion","quotes":["verbatim supporting text"],
"why_useful":"connection to the real query"}]}. If none is useful, return
{"answers":[]}. Do not repeat conclusions listed in already_studied.
Quotes locate source text; do not claim independent verification.'''

# OLMo-3 almost always takes the empty-answer escape of the original prompt
# (0/6 LongHealth groups) and echoes the literal JSON placeholder. Same task,
# no escape, explicit placeholders (A/B: runs/olmo_dev/prompt*.log). Opt-in only.
ANSWER_INSTRUCTION_NONEMPTY = ANSWER_INSTRUCTION.replace(
    'If none is useful, return\n{"answers":[]}.',
    'Always return at least one conclusion that the evidence supports and that\nrelates to the query (it may help only partially).').replace(
    '{"answers":[{"answer":"conclusion","quotes":["verbatim supporting text"],\n"why_useful":"connection to the real query"}]}',
    '{"answers":[{"answer":"<a concrete conclusion stated in full>","quotes":["<verbatim supporting text>"],\n"why_useful":"<connection to the real query>"}]}\n(replace every <...> placeholder with real content)')
assert ANSWER_INSTRUCTION_NONEMPTY.count('<') >= 3 and 'If none' not in ANSWER_INSTRUCTION_NONEMPTY

QUESTION_INSTRUCTION = '''Construct one self-contained closed-book study question
for each supplied conclusion. Its answer must be that conclusion. The question
must help answer the real query or learn its prerequisites, preserve conditions
and negation, and require the joint information when its answer combines sources.
Do not state the answer or all solution facts in the question. Do not refer to
"the passage above" or source labels. Return JSON only:
{"questions":[{"answer_index":0,"question":"..."}, ...]}.
Do not copy the real query or repeat any already_studied question.'''


async def generate_pair(engine, components, config, row, case, deadline):
    from cartridges.structs import Conversation, read_conversations
    from cartridges.synthesizers.self_study import SYSTEM_PROMPT_TEMPLATE
    from experiments.build_synthetic_pair_fast import synthesize_original
    from experiments.vllm_study_client import sampling_seed
    from contextualized.answer_guided_questions import parse_object, closed_book_question_rejection, literal_quotes
    root, files = paths(config, row)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(row['synthesis_manifest'])
    if manifest_path.exists():
        validate_pair_manifest(row, json.loads(manifest_path.read_text()))
        return
    case = public_case(case)
    sources = json.loads((root / 'sources.json').read_text())
    client = components.client(components.client.Config(target=components.client,
        model_name=engine.tokenizer.name_or_path, runtime_key=components.runtime_key))
    required = config.get('synthetic_questions_per_request_per_arm', 30)
    if type(required) is not int or required < 1:
        raise ValueError('A positive minimum question count is required')
    seed = config['seed'] + 10000 * row['request_index']
    if config.get('reuse_existing_questions'):
        from experiments.reuse_synthesis_questions import seed_existing_pair
        await seed_existing_pair(engine, client, config, row, files, deadline)
    started = time.monotonic()
    call_log = root / 'calls.jsonl'

    def check_time():
        if time.time() >= deadline - 30:
            raise TimeoutError('Synthetic generation reached the allocation deadline')

    async def text_call(instruction, payload, request_seed):
        check_time()
        with sampling_seed(request_seed):
            response = await client.chat([[{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]],
                temperature=.6, max_completion_tokens=1024, top_logprobs=None)
        sample = response.samples[0]
        with call_log.open('a') as handle:
            handle.write(json.dumps(dict(instruction=instruction, payload=payload,
                response=sample.text, complete=finished(sample, engine.eos_ids),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens), ensure_ascii=False)+'\n')
        return parse_object(sample.text) if finished(sample, engine.eos_ids) else None

    answer_instruction = (ANSWER_INSTRUCTION_NONEMPTY if config.get('answer_instruction') == 'nonempty_v2'
                          else ANSWER_INSTRUCTION)

    async def make_group(group, attempt, previous):
        response = await text_call(answer_instruction, dict(query=case['prompt'],
            evidence=group['text'], already_studied=[e['answer'] for e in previous]), seed+attempt*100)
        answers = response.get('answers', []) if isinstance(response, dict) else []
        answers = [a for a in answers if isinstance(a, dict) and isinstance(a.get('answer'), str)
                   and a['answer'].strip() and not re.fullmatch(r'<[^>]*>|conclusion', a['answer'].strip(), re.I)][:3]
        if not answers:
            return []
        response = await text_call(QUESTION_INSTRUCTION, dict(query=case['prompt'],
            evidence=group['text'], conclusions=answers,
            already_studied=[e['question'] for e in previous]), seed+attempt*100+1)
        questions = response.get('questions', []) if isinstance(response, dict) else []
        if not isinstance(questions, list):
            return []
        output = []
        for value in questions:
            if (isinstance(value, dict) and type(value.get('answer_index')) is not int
                    and isinstance(value.get('answer'), str) and config.get('answer_instruction') == 'nonempty_v2'):
                # OLMo-3 sometimes names the conclusion instead of its index; map an exact match only.
                matches = [i for i, a in enumerate(answers) if key(a['answer']) == key(value['answer'])]
                if len(matches) == 1:
                    value = dict(value, answer_index=matches[0])
            if not isinstance(value, dict) or type(value.get('answer_index')) is not int:
                continue
            index, question = value['answer_index'], value.get('question')
            if not 0 <= index < len(answers) or not isinstance(question, str) or not question.strip():
                continue
            anchor = answers[index]
            quotes = literal_quotes(anchor.get('quotes'), group['source_chunks']) or []
            if (key(question) in (key(case['query']), key(case['prompt']))
                    or closed_book_question_rejection(question, anchor['answer'], [q['quote'] for q in quotes])):
                continue
            output.append(dict(question=question.strip(), answer=anchor['answer'].strip(),
                evidence=group, answer_source_quotes=quotes,
                raw_answer_source_quotes=anchor.get('quotes'), why_useful=anchor.get('why_useful'),
                source_facts_verified=False, joint_source_support_verified=False,
                answer_used_as_SD_label=False, origin=VERSION))
        return output

    specific_file = Path(files['specific_json'])
    examples = json.loads(specific_file.read_text()).get('examples', []) if specific_file.exists() else []
    specific = read_conversations(files['specific_targets']) if Path(files['specific_targets']).exists() else []
    # A crash between the two atomic writes can leave uncommitted proposals,
    # but never creates duplicate training rows or treats partial data as ready.
    examples = examples[:len(specific)]
    if len(specific) > len(examples):
        raise ValueError('Specific soft targets have no matching question provenance')
    state_path = root/'generation_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'specific_attempt': 0, 'agnostic_attempt': 0}
    groups = sources['evidence']
    if not groups:
        raise ValueError('Reranker returned no source groups')
    seen = {key(e['question']) for e in examples}
    # Attempts are checkpointed. A shortfall is explicit, never padded with q.
    while len(examples) < required and state['specific_attempt'] < config['max_evidence_attempts']:
        check_time()
        first = state['specific_attempt']
        stop = min(first+4, config['max_evidence_attempts'])
        proposed = await asyncio.gather(*(make_group(groups[i % len(groups)], i, examples)
                                         for i in range(first, stop)))
        candidates = []
        for batch in proposed:
            for item in batch:
                if key(item['question']) not in seen:
                    seen.add(key(item['question'])); candidates.append(item)
        candidates = candidates[:required-len(examples)]
        if candidates:
            chats = [[{'role':'system','content':SYSTEM_PROMPT_TEMPLATE.format(subcorpus=e['evidence']['text'])},
                      {'role':'user','content':e['question']}] for e in candidates]
            with sampling_seed(seed+first*100+2):
                teacher = await client.chat(chats, temperature=0., max_completion_tokens=1024, top_logprobs=20)
            for e, chat, answer in zip(candidates, chats, teacher.samples):
                if not finished(answer, engine.eos_ids):
                    continue
                specific.append(Conversation(system_prompt=chat[0]['content'], messages=[
                    Conversation.Message(content=e['question'], role='user', token_ids=None),
                    Conversation.Message(content=answer.text, role='assistant', token_ids=answer.token_ids,
                        top_logprobs=answer.top_logprobs.flatten(.99))],
                    metadata=dict(source=VERSION, question_index=len(examples), query_condition=case['prompt'],
                        benchmark_answer_used=False, anchor_answer_is_training_label=False,
                        complete_teacher_answer=True), type='self_study'))
                examples.append(dict(e, teacher_answer=answer.text))
        state['specific_attempt'] = stop
        write(specific_file, dict(version=VERSION, case=row['case'], dataset=row['dataset'],
            query=case['prompt'], examples=examples, requested_examples=required, complete=len(examples)>=required,
            original_query_teacher_fallback=False, source_text_abridged=False, benchmark_answer_used=False))
        save_targets(specific, files['specific_targets'])
        write(state_path, state)
    if len(examples) < required:
        write(root/'shortfall.json', dict(arm='specific', accepted=len(examples), required=required,
            attempts=state['specific_attempt'], duplicate_query_padding=False))
        raise ValueError(f"Specific synthesis shortfall: {len(examples)}/{required}; see {root}")

    # The original arm never receives q, the extracted anchors, or retrieval.
    agnostic_file = Path(files['agnostic_json'])
    agnostic = read_conversations(files['agnostic_targets']) if Path(files['agnostic_targets']).exists() else []
    # Documents recur across requests. Tokenize/chunk each document once per
    # worker; the sampled contexts, prompts and per-request RNG stay unchanged.
    if not hasattr(engine, 'agnostic_context_cache'):
        engine.agnostic_context_cache = {}
    if row['document_sha256'] not in engine.agnostic_context_cache:
        ids = engine.tokenizer.encode(case['document'], add_special_tokens=False)
        engine.agnostic_context_cache[row['document_sha256']] = [
            engine.tokenizer.decode(ids[i:i+2048]) for i in range(0, len(ids), 2048)]
    contexts = engine.agnostic_context_cache[row['document_sha256']]
    while len(agnostic) < required and state['agnostic_attempt'] < 12:
        check_time()
        new = await synthesize_original(engine, components, contexts, count=required-len(agnostic),
            seed=seed+500000+state['agnostic_attempt'], question_tokens=512, answer_tokens=1024,
            batch_size=8, exclude={key(c.messages[0].content) for c in agnostic}, deadline_epoch=deadline-30)
        agnostic += [c for c in new if valid_conversation(c, engine.eos_ids)]
        state['agnostic_attempt'] += 1
        save_targets(agnostic, files['agnostic_targets'])
        write(agnostic_file, dict(version=VERSION, case=row['case'], dataset=row['dataset'],
            query_conditioned=False, benchmark_answer_used=False, requested_examples=required,
            synthesizer='cartridges.synthesizers.self_study.SelfStudySynthesizer',
            question_max_tokens=512, teacher_answer_max_tokens=1024, complete=len(agnostic)>=required,
            questions=[dict(question=c.messages[0].content, answer=c.messages[-1].content,
                source_system_prompt=c.system_prompt) for c in agnostic]))
        write(state_path, state)
    if len(agnostic) < required:
        raise ValueError(f'Agnostic synthesis shortfall: {len(agnostic)}/{required}')
    if not all(valid_conversation(c, engine.eos_ids) for c in specific+agnostic):
        raise ValueError('Incomplete Teacher targets cannot enter the matched training set')
    write(manifest_path, dict(version=VERSION, complete=True, count_per_arm=min(len(specific),len(agnostic)),
        minimum_per_arm=required,actual_counts=dict(specific=len(specific),agnostic=len(agnostic)),
        **{k:row[k] for k in ('case','dataset','document_sha256','prompt_sha256')},
        data_files=files, data_sha256={k:sha_file(p) for k,p in files.items()},
        resumed_segment_seconds=time.monotonic()-started, benchmark_answers_used=False,
        absolute_quality_verified=False, original_query_teacher_fallback=False))


def generate(config, rows, cases, deadline):
    from experiments.vllm_study_client import load_vllm_runtime
    from experiments.build_synthetic_pair_fast import make_runtime_components
    engine = load_vllm_runtime(config['model'], seed=config['seed'],
        max_batch_sequences=config.get('generation_max_sequences', 16),
        max_model_len=16384, gpu_memory_utilization=config.get('generation_gpu_memory_utilization', .75),
        max_num_batched_tokens=8192)
    components = make_runtime_components(engine)
    parallel = config.get('generation_parallel_requests', 2)
    if type(parallel) is not int or not 1 <= parallel <= 8:
        raise ValueError('generation_parallel_requests must be in [1, 8]')
    async def one(row):
        from experiments.evidence_only_thirty import generate_evidence_only, report_progress
        case = cases[row['dataset']][row['case']]
        tasks = [('pair', generate_pair(engine, components, config, row, case, deadline))]
        if config.get('include_evidence_only', False):
            tasks.append(('evidence_only', generate_evidence_only(
                engine, components, config, row, case, deadline)))
        results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        for (arm, _), result in zip(tasks, results):
            if isinstance(result, TimeoutError):
                raise result
            if isinstance(result, Exception):
                import traceback
                write(Path(config['output'])/'failures'/f"synthesis-{arm}-{row['dataset']}-{row['case']}.json",
                    dict(error=str(result), traceback=''.join(traceback.format_exception(result)),
                         dataset=row['dataset'], case=row['case'], arm=arm))
        report_progress(config, rows)
        print(json.dumps(dict(event='synthesis_request_finished', dataset=row['dataset'],
            case=row['case'], errors=[str(x) for x in results if isinstance(x, Exception)])), flush=True)
    async def run():
        # A bounded queue keeps the GPU supplied while a slower request finishes.
        semaphore = asyncio.Semaphore(parallel)
        async def limited(row):
            async with semaphore:
                if time.time() >= deadline-30:
                    raise TimeoutError('Synthesis deadline')
                await one(row)
        await asyncio.gather(*(limited(row) for row in rows))
        await engine.batcher.wait_idle()
    asyncio.run(run())
