"""Frozen source-conditioned Teacher targets and explicitly configured selective SD."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
import json
import os
from pathlib import Path
import time
import zlib

from experiments.run_syn_sd_factorial import (read, write, input_rows,
    materialize_training_prompts, critic_task_prompt,
    save_conversation, load_conversation)


def reuse_complete_preparation(case, inputs, config, output):
    """Reuse identical frozen supervision without starting a GPU Teacher."""
    if config.get('selective_gate') not in {
            'verified_quote_grounded_veto','complete_answer_verified_quote_grounded_veto',
            'complete_answer_verified_quote_grounded_veto_v2',
            'answer_preserving_verified_quote_grounded_veto_v3',
            'verified_quote_support_required_v4',
            'task_conditioned_support_required',
            'original_task_conditioned_support_required'}:
        return False
    if not (output/'baselines.json').is_file():
        return False
    for dataset, splits in inputs.items():
        for split, rows in splits.items():
            for index in range(len(rows)):
                if not (output/'targets'/dataset/split/f'{index:03d}.pkl').is_file():
                    return False
                if not (output/'assessments'/dataset/split/f'{index:03d}.json').is_file():
                    return False
    from experiments.selective_learning_runtime import SYSTEM_TEMPLATE
    audits = {}
    reused = 0
    for dataset, splits in inputs.items():
        audits[dataset] = {'train': [], 'validation': []}
        for split, rows in splits.items():
            for index, row in enumerate(rows):
                conv = load_conversation(output/'targets'/dataset/split/f'{index:03d}.pkl',
                                         row, dataset, split)
                context = (row['evidence'] if config.get('teacher_context_policy')=='question_source_evidence'
                           else case['document'])
                temperature_mismatch = (
                    not config.get('fixed_synthetic_answer_trajectories', False)
                    and conv.metadata.get('teacher_sampling_temperature') !=
                        config['teacher_sampling_temperature'])
                if (conv.system_prompt != SYSTEM_TEMPLATE.format(subcorpus=context)
                        or temperature_mismatch
                        or conv.messages[-1].top_logprobs is None):
                    raise ValueError('Cached Teacher source/temperature/soft targets differ')
                reused += 1
                path = output/'assessments'/dataset/split/f'{index:03d}.json'
                audit = read(path)
                if (audit['question'] != row['question']
                        or audit['candidate'] != conv.messages[-1].content
                        or audit.get('critic_context_policy') != config.get('selective_critic_context')
                        or audit['assessment'].get('decision_source') != config['selective_gate']):
                    raise ValueError('Cached admission is from a different source/gate protocol')
                audits[dataset][split].append({'question_index':index,
                    'apply_update':audit['assessment']['apply_update'],
                    'reason':audit['assessment']['reason'],'path':str(path)})
    write(output/'supervision_summary.json', {
        'target_cache':{'new':0,'reused':reused},
        'train_counts':{dataset:{'vanilla_sd':len(splits['train']),
            'selective_sd':sum(a['apply_update'] for a in audits[dataset]['train'])}
            for dataset,splits in inputs.items()},
        'validation_counts':{dataset:{'vanilla_sd':len(splits['validation']),
            'selective_sd':sum(a['apply_update'] for a in audits[dataset]['validation'])}
            for dataset,splits in inputs.items()},
        'gate':config['selective_gate'],
        'gate_audits':{dataset:value['train'] for dataset,value in audits.items()},
        'validation_gate_audits':{dataset:value['validation'] for dataset,value in audits.items()},
        'transport':{'backend':'cached_vllm_raw_top20',
            'prepared_cache_reused_without_loading_teacher':True,
            'teacher_context_policy':config.get('teacher_context_policy'),
            'teacher_sampling_temperature':config['teacher_sampling_temperature'],
            'selective_critic_context':config.get('selective_critic_context')},
        'heldout_policy':config['validation_policy'],'teacher_targets_unchanged':True})
    print(json.dumps({'event':'complete_frozen_supervision_reused',
        'case':case['key'],'targets':reused,'Teacher_GPU_loaded':False}),flush=True)
    return True


def prepare_full_context_baselines(case,config,output,tokenizer,deadline):
    """Evaluate the full-document comparator without enlarging the study engine."""
    path=output/'baselines.json'
    if path.is_file():return
    if time.time()>=deadline:raise TimeoutError('Full-context baseline deadline')
    import torch
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    from contextualized.data import task_without_document
    from experiments.selective_learning_runtime import FrozenAssessmentGenerator
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id};eos.discard(None)
    engine=SimpleNamespace(teacher=model,tokenizer=tokenizer,device=torch.device('cuda'),eos_ids=eos)
    generate=FrozenAssessmentGenerator(engine,prefill_tokens=1024)
    full=generate(case['prompt'],config['eval_tokens'])
    bare=generate(task_without_document(case),config['eval_tokens'])
    write(path,{'full_context_teacher':full['prediction'],'no_context':bare['prediction'],
        'full_teacher_prompt':'original_benchmark_task_and_full_document',
        'full_teacher_complete':full['complete'],'no_context_complete':bare['complete'],
        'teacher_backend':{'name':'huggingface_sdpa','prefill_tokens':1024},
        'benchmark_answers_used':False})
    del generate,engine,model;gc.collect();torch.cuda.empty_cache()


async def prepare(runtime, components, case, inputs, config, output, seed, deadline):
    from cartridges.structs import Conversation
    from experiments.vllm_study_client import sampling_seed
    from experiments.selective_learning_runtime import assess_teacher_signal, _legacy_helpers, SYSTEM_TEMPLATE
    client = components.client(components.client.Config(target=components.client,
        model_name=runtime.tokenizer.name_or_path, runtime_key=components.runtime_key))
    loop = asyncio.get_running_loop(); legacy = _legacy_helpers(config['stack_ttt_root'])
    stats = {'new': 0, 'reused': 0}; targets = {}
    source_conditioned=config.get('teacher_context_policy')=='question_source_evidence'
    def source_context(row):return row['evidence'] if source_conditioned else case['document']

    async def infer(chats, *, tokens, temperature=0., logprobs=None, local_seed=0):
        if time.time() >= deadline: raise TimeoutError('Frozen Teacher deadline')
        with sampling_seed((seed+local_seed) % (2**31)):
            response = await client.chat(chats, temperature=temperature,
                max_completion_tokens=tokens, top_logprobs=logprobs, enable_thinking=False)
        return response

    baseline = output/'baselines.json'
    if not baseline.is_file():
        from contextualized.data import task_without_document
        response = await infer([
            [{'role':'user','content':case['prompt']}],
            [{'role':'user','content':task_without_document(case)}]], tokens=config['eval_tokens'])
        values = response.samples
        write(baseline, {'full_context_teacher': values[0].text, 'no_context': values[1].text,
            'full_teacher_prompt': 'original_benchmark_task_and_full_document',
            'teacher_backend': runtime.backend_metadata,
            'full_teacher_complete': values[0].token_ids[-1] in runtime.eos_ids,
            'no_context_complete': values[1].token_ids[-1] in runtime.eos_ids})

    async def target(dataset, split, index, row):
        path = output/'targets'/dataset/split/f'{index:03d}.pkl'
        if path.is_file():
            stats['reused'] += 1
            return load_conversation(path, row, dataset, split)
        began = time.monotonic()
        context=source_context(row)
        prompt=row.get('training_prompt',row['question'])
        chat = [{'role':'system','content':SYSTEM_TEMPLATE.format(subcorpus=context)},
                {'role':'user','content':prompt}]
        fixed = config.get('fixed_synthetic_answer_trajectories', False)
        if fixed:
            answer_ids = row.get('answer_token_ids')
            if answer_ids is None:
                answer_ids = runtime.tokenizer.encode(row['answer'], add_special_tokens=False)
                if not answer_ids:
                    raise ValueError('Tokenizer produced an empty fixed answer trajectory')
                if row.get('answer_finished', True) and answer_ids[-1] not in runtime.eos_ids:
                    eos = runtime.tokenizer.eos_token_id
                    if eos is None or eos not in runtime.eos_ids:
                        raise ValueError('Cannot terminate a fixed synthetic answer trajectory')
                    answer_ids.append(int(eos))
            answer_ids = list(map(int, answer_ids))
            if len(answer_ids) > config['learning_answer_tokens']:
                raise ValueError('Saved assistant trajectory exceeds the frozen answer budget')
            with sampling_seed((seed+1000+100*(split == 'validation')+index) % (2**31)):
                response = await client.teacher_force([chat], [answer_ids],
                    top_logprobs=config['teacher_topk'], enable_thinking=False)
        else:
            response = await infer([chat], tokens=config['learning_answer_tokens'],
                temperature=config['teacher_sampling_temperature'], logprobs=config['teacher_topk'],
                local_seed=1000+100*(split == 'validation')+index)
        sample = response.samples[0]
        if sample.top_logprobs is None: raise ValueError('Missing real Teacher soft targets')
        conversation = Conversation(messages=[
            Conversation.Message(role='user', content=prompt,
                token_ids=runtime.tokenizer.encode(prompt, add_special_tokens=False)),
            Conversation.Message(role='assistant', content=sample.text,
                token_ids=list(sample.token_ids), top_logprobs=sample.top_logprobs.flatten(.99))],
            system_prompt=SYSTEM_TEMPLATE.format(subcorpus=context),
            metadata={'question_index':index,'original_question':row['question'],
                'dataset_arm':dataset,'split':split,'teacher_context':config.get('teacher_context_policy','full_document'),
                'supervision':'unchanged_source_Teacher_sparse_SD',
                'teacher_sampling_temperature':config['teacher_sampling_temperature'],
                'teacher_finished':sample.token_ids[-1] in runtime.eos_ids,
                'source_row_index':row['source_row_index'],'synthetic_answer_labels_used':False,
                'synthetic_answer_trajectory_used':fixed,
                'synthetic_answer_hard_label_loss':False,
                'answer_trajectory_source':row.get('answer_trajectory_source'),
                'training_prompt_policy':row.get('training_prompt_policy'),
                'training_prompt_sha256':row.get('training_prompt_sha256'),
                'generation_seconds':time.monotonic()-began,'teacher_backend':'vllm_raw_top20',
                'benchmark_answers_used':False}, type='query_guided_standard_SD')
        save_conversation(conversation,path); stats['new'] += 1
        print(json.dumps({'event':'batched_teacher_target','case':case['key'],
            'dataset':dataset,'split':split,'index':index,'answer_tokens':len(sample.token_ids)}),flush=True)
        return conversation

    for dataset,splits in inputs.items():
        targets[dataset] = {}
        jobs = [(split,index,row) for split,rows in splits.items() for index,row in enumerate(rows)]
        values = await asyncio.gather(*(target(dataset,split,index,row) for split,index,row in jobs))
        for split in splits:
            targets[dataset][split] = [value for (s,_,_),value in zip(jobs,values) if s == split]

    async def assessment(content, budget):
        response = await infer([[{'role':'user','content':content}]],tokens=budget)
        sample = response.samples[0]; complete = sample.token_ids[-1] in runtime.eos_ids
        return {'prediction':sample.text,'generated_token_ids':list(sample.token_ids),
            'prompt_tokens':response.usage.prompt_tokens,'complete':complete,
            'finish_reason':'eos' if complete else 'length'}

    # A separate pool prevents blocking critics from starving vLLM's engine thread.
    pool = ThreadPoolExecutor(max_workers=config.get('critic_concurrency',16),thread_name_prefix='source_gate')
    audits = {}
    async def gate(dataset,split,index,row,conversation):
        path = output/'assessments'/dataset/split/f'{index:03d}.json'
        candidate = conversation.messages[-1].content
        if path.is_file():
            value = read(path)
            if value['question'] != row['question'] or value['candidate'] != candidate:
                raise ValueError('Stored selective decision differs from its unchanged Teacher target')
            if config.get('selective_critic_context') and value.get('critic_context_policy')!=config['selective_critic_context']:
                raise ValueError('Cached admission used a different critic context')
        else:
            began = time.monotonic()
            def bridge(content,budget):
                future = asyncio.run_coroutine_threadsafe(assessment(content,budget),loop)
                return future.result(timeout=max(1,deadline-time.time()))
            document=(case['document'] if config.get('selective_critic_context')=='full_document_with_focused_source_quotes'
                      else source_context(row))
            task_prompt = critic_task_prompt(case,row,dataset,config,document)
            result = await loop.run_in_executor(pool,partial(assess_teacher_signal,
                document=document,question=row['question'],task_prompt=task_prompt,
                evidence=row['evidence'],candidate=candidate,
                candidate_complete=conversation.metadata['teacher_finished'],
                generate_assessment=bridge,legacy_helpers=legacy,
                admission_policy=config.get('selective_gate','task_conditioned_support_required')))
            value = {'question_index':index,'question':row['question'],'candidate':candidate,
                'dataset':dataset,'split':split,'assessment':result,'teacher_targets_unchanged':True,
                'critic_context_policy':config.get('selective_critic_context','teacher_source'),
                'critic_source_characters':len(document),
                'seconds':time.monotonic()-began,'benchmark_answers_used':False,
                'synthetic_answer_labels_used':False}
            write(path,value)
        print(json.dumps({'event':'batched_selective_signal','case':case['key'],
            'dataset':dataset,'split':split,'index':index,
            'accepted':value['assessment']['apply_update']}),flush=True)
        return {'question_index':index,'apply_update':value['assessment']['apply_update'],
                'reason':value['assessment']['reason'],'path':str(path)}
    try:
        for dataset,splits in inputs.items():
            audits[dataset] = {}
            for split in ('train','validation'):
                audits[dataset][split] = await asyncio.gather(*(gate(dataset,split,index,row,conversation)
                    for index,(row,conversation) in enumerate(zip(
                        splits[split],targets[dataset][split]))))
    finally: pool.shutdown(wait=True)
    await runtime.batcher.wait_idle()
    write(output/'supervision_summary.json',{'target_cache':stats,
        'train_counts':{dataset:{'vanilla_sd':len(targets[dataset]['train']),
            'selective_sd':sum(row['apply_update'] for row in audits[dataset]['train'])} for dataset in inputs},
        'validation_counts':{dataset:{'vanilla_sd':len(targets[dataset]['validation']),
            'selective_sd':sum(row['apply_update'] for row in audits[dataset]['validation'])} for dataset in inputs},
        'gate':config['selective_gate'],
        'gate_audits':{dataset:value['train'] for dataset,value in audits.items()},
        'validation_gate_audits':{dataset:value['validation'] for dataset,value in audits.items()},
        'transport':{'backend':runtime.backend_metadata,'concurrent_source_critics':True,
            'legacy_grounded_veto_prompts_reused':True,
            'teacher_context_policy':config.get('teacher_context_policy','full_document'),
            'teacher_sampling_temperature':config['teacher_sampling_temperature']},
        'heldout_policy':config['validation_policy'],'teacher_targets_unchanged':True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--case',required=True)
    a = p.parse_args();config = read(a.config)
    item = next(row for row in config['cases'] if row['key'] == a.case)
    from contextualized.data import read_indexed_case
    plan = read(config['source_plan']); index = {row['key']:row for row in read(plan['source_index'])}
    case = read_indexed_case(plan,index,a.case);output=Path(config['output'])/'cases'/a.case
    seed=(config['seed']+zlib.crc32(a.case.encode())) % (2**31)
    inputs,_=input_rows(item,config,seed)
    materialize_training_prompts(case,inputs,config)
    allocation_deadline=float(os.environ.get(
        'PAIR_DEADLINE_EPOCH',config['deadline_epoch']))
    deadline=(allocation_deadline if config.get('deadline_from_slurm_allocation')
              else min(config['deadline_epoch'],allocation_deadline))
    if reuse_complete_preparation(case,inputs,config,output):
        return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    if config.get('baseline_backend')=='hf_separate':
        prepare_full_context_baselines(case,config,output,tokenizer,deadline)
        context_tokens=config.get('study_max_model_len',16384)
    else:context_tokens=max(config.get('study_max_model_len',16384),
        len(tokenizer.encode(case['document'],add_special_tokens=False))+8192)
    from experiments.vllm_study_client import load_vllm_runtime
    from experiments.build_synthetic_pair_fast import make_runtime_components
    # Prompt-logprob materialization temporarily allocates a full-vocabulary
    # tensor.  Leave extra headroom on 24 GiB workers; this changes only the
    # transport batch size/KV-cache reservation, never prompts or targets.
    import torch
    device_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch_sequences = config.get('teacher_max_batch_sequences',32)
    memory_utilization = config.get('teacher_gpu_memory_utilization',.85)
    if device_gib < 32:
        batch_sequences = min(batch_sequences, 4)
        memory_utilization = min(memory_utilization, .60)
    runtime=load_vllm_runtime(config['model'],tokenizer=tokenizer,seed=seed,
        max_model_len=context_tokens,max_batch_sequences=batch_sequences,
        max_num_batched_tokens=config.get('teacher_max_num_batched_tokens',4096),
        gpu_memory_utilization=memory_utilization,
        enforce_eager=config.get('teacher_enforce_eager',False))
    components=make_runtime_components(runtime)
    asyncio.run(prepare(runtime,components,case,inputs,config,output,seed,deadline))
    from contextualized.upstream_study import _RUNTIMES
    _RUNTIMES.pop(components.runtime_key,None)


if __name__ == '__main__':main()
