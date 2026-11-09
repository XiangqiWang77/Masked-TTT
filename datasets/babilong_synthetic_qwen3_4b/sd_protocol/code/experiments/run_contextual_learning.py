"""Full native-KV SD comparison with source-supported selective learning.

This runner never loads benchmark references and performs no hash or smoke checks.
Explicit legacy configurations retain the earlier counterfactual experiment.
"""
import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def learning_protocol(config):
    """Keep historical v1/v2 runs explicit; new configurations use selective SD."""
    protocol=config.get('learning_protocol')
    if protocol is None:
        protocol=('legacy_counterfactual' if config.get('protocol') in
                  {'contextual-complete-v1','contextual-complete-v2'} else 'selective_sd')
    if protocol not in {'selective_sd','legacy_counterfactual'}:
        raise ValueError('learning_protocol must be selective_sd or legacy_counterfactual')
    return protocol


def guided_questions(path):
    """Read either question-only rows or the old Answer-Guided examples.

    Only question/evidence fields are passed downstream. In particular, a
    crafted answer is never converted into a hard Student training label.
    """
    value=read(path)
    if isinstance(value,dict) and isinstance(value.get('content'),dict):
        value=value['content']
    rows=value.get('questions',value.get('examples')) if isinstance(value,dict) else value
    if not isinstance(rows,list): raise ValueError('Missing guided question/example rows')
    return [{'question':row['question'],'evidence':row['evidence']} for row in rows]


def load_engine(config):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from cartridges.cache import AttnConfig
    from cartridges.models import FlexQwen3ForCausalLM
    from contextualized.engine import CartridgeEngine
    kernel_options={}
    if config.get('generation_small_tiles',False):
        from experiments.flex_kernel_compat import install_generation_small_tiles
        kernel_options=install_generation_small_tiles()
    # These existing engine methods do not need the old independent verifier.
    engine=object.__new__(CartridgeEngine)
    engine.device=torch.device('cuda');engine.prefix_tokens=config['prefix_tokens'];engine.packed_tokens=config['packed_tokens']
    engine.tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    engine.teacher=AutoModelForCausalLM.from_pretrained(config['model'],torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).to(engine.device).eval().requires_grad_(False)
    engine.student=FlexQwen3ForCausalLM.from_pretrained(config['model'],torch_dtype=torch.bfloat16,
        local_files_only=True).to(engine.device).eval().requires_grad_(False)
    eos=engine.teacher.generation_config.eos_token_id
    engine.eos_ids=set(eos if isinstance(eos,list) else [eos])|{engine.tokenizer.eos_token_id}
    engine.eos_ids.discard(None)
    c=engine.student.config;engine.attn=AttnConfig(c.num_hidden_layers,c.num_key_value_heads,c.head_dim)
    engine.generation_kernel_options=kernel_options
    return engine


def train_arm(engine, initial, conversations, validation, config, seed):
    from contextualized.early_study import fit_early,StopRule,fixed_update_rule
    cache=engine.clone_cache(initial)
    if not conversations:
        return cache,{'updates':0,'stop_reason':'no_eligible_training_conversations',
            'objective':'upstream_sparse_teacher_distillation','direct_answer_nll':False,
            'train_conversations':0,'heldout_conversations':len(validation),'fallback_supervision_used':False}
    rule=(fixed_update_rule(config['fixed_updates'],min_steps=config['min_steps'],eval_every=config['eval_every'],
            track_best_before_min=config.get('track_best_before_min',False))
        if config.get('fixed_updates') else StopRule(min_steps=config['min_steps'],max_steps=config['max_steps'],
            eval_every=config['eval_every'],patience=config['patience'],
            track_best_before_min=config.get('track_best_before_min',False)))
    history=fit_early(engine,cache,conversations,validation,lr=config['learning_rate'],
        batch_size=config['train_batch_size'],packed_tokens=config['packed_tokens'],seed=seed,
        rule=rule)
    return cache,history


def run(config, rank):
    import torch
    from cartridges.structs import read_conversations
    from contextualized.data import read_indexed_case,task_without_document
    from contextualized.cache_io import save_cartridge
    protocol=learning_protocol(config)
    if protocol=='selective_sd':
        from experiments.selective_learning_runtime import prepare_selective_conversations
        audit_field='selective_sd'
    else:
        from experiments.contextual_learning_runtime import prepare_cf_conversations
        audit_field='counterfactual'
    root=Path(config['output']);plan=read(config['plan']);index={r['key']:r for r in read(plan['source_index'])}
    if not 0<=rank<config['workers']<=12: raise ValueError('Invalid GPU rank')
    assigned=plan['cases'][rank::config['workers']]
    deadline=min(float(config['deadline_epoch']),float(os.environ.get('PAIR_DEADLINE_EPOCH',config['deadline_epoch'])))
    if time.time()>=deadline: return 75
    engine=load_engine(config)
    failures=[];done=[]
    for position,item in enumerate(assigned):
        key=item['key'];dest=root/'results/cases'/(key+'.json')
        if dest.exists():
            existing=read(dest)
            if existing.get('case')!=key or existing.get('status')!='completed': raise ValueError('Invalid saved result')
            existing_protocol=existing.get('learning_protocol',
                'legacy_counterfactual' if 'counterfactual' in existing else None)
            if existing_protocol!=protocol: raise ValueError('Saved result uses a different learning protocol')
            done.append(key);continue
        if time.time()>=deadline: break
        started=time.monotonic();seed=config['seed']+rank+position*config['workers']
        random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        print(json.dumps({'event':'learning_case_started','case':key,'rank':rank}),flush=True)
        try:
            case=read_indexed_case(plan,index,key);record=read(root/'dataset_cases'/(key+'.json'))
            if not record.get('complete') or record['case']!=key: raise ValueError('Dataset case is incomplete')
            paths=record['paths'];original_train=read_conversations(paths['original_train'])
            original_val=read_conversations(paths['original_validation'])
            guided=guided_questions(paths['guided']);source_record=read(paths['source'])
            if isinstance(source_record.get('content'),dict): source_record=source_record['content']
            sources=source_record['chunks']
            evidence_core=sorted({int(c['source_index'])
                for group in source_record.get('selected_evidence',[]) for c in group['source_chunks']})
            if protocol=='legacy_counterfactual' and config.get('preserve_evidence_core',False) and not evidence_core:
                raise ValueError('Legacy core-preserving CF requires the original selected-evidence core')
            if not original_train: raise ValueError('Missing original SD conversations')
            query=task_without_document(case);timings={};predictions={};training={}
            start=time.monotonic();full=engine.generate_teacher(query,case['document'],max_new_tokens=config['eval_tokens'])
            predictions['full_context']=engine.text(full);timings['full_context_seconds']=time.monotonic()-start
            del full
            evidence_indices={int(c['source_index']) for row in guided for c in row['evidence']['source_chunks']}
            if protocol=='legacy_counterfactual' and config.get('preserve_evidence_core',False): evidence_indices=set(evidence_core)
            evidence='\n\n'.join(s['text'] for s in sorted(sources,key=lambda s:s['span'][0])
                                  if int(s['source_index']) in evidence_indices)
            if evidence:
                local=engine.generate_teacher(query,evidence,max_new_tokens=config['eval_tokens'])
                predictions['evidence_teacher']=engine.text(local);del local
            else: predictions['evidence_teacher']=''
            initial=engine.initial_cache(case['document'])
            start=time.monotonic();vanilla,training['vanilla_cartridge']=train_arm(engine,initial,original_train,original_val,config,seed)
            predictions['vanilla_cartridge']=engine.generate_student(query,vanilla,max_new_tokens=config['eval_tokens'])[0]
            vanilla.clear();timings['vanilla_train_seconds']=time.monotonic()-start
            if config.get('save_cartridges',True): save_cartridge(vanilla,root/'cartridges'/key/'vanilla.safetensors',config['model'])
            del vanilla;gc.collect();torch.cuda.empty_cache()
            start=time.monotonic()
            if protocol=='selective_sd':
                def task_prompt_builder(document,question):
                    # Preserve the source task instructions/formatting and
                    # change only its question, without reading a reference.
                    return task_without_document(case,question).replace(
                        '[The context is provided above.]',document,1)
                prepared=asyncio.run(prepare_selective_conversations(engine,initial,guided,sources,
                    document=case['document'],seed=seed,answer_tokens=config['learning_answer_tokens'],
                    deadline_epoch=deadline,concise_answers=config.get('concise_answers',False),
                    stack_ttt_root=config.get('stack_ttt_root','/home/da839/Stack_TTT'),
                    task_prompt_builder=task_prompt_builder))
                prepared['audit']['task_prompt_policy']='original_task_with_current_synthetic_question'
                selected_data=prepared['selective_conversations']
            else:
                prepared=asyncio.run(prepare_cf_conversations(engine,initial,guided,sources,
                    document=case['document'],seed=seed,answer_tokens=config['learning_answer_tokens'],
                    backgrounds=config['counterfactual_backgrounds'],deadline_epoch=deadline,
                    evidence_core_source_indices=evidence_core,
                    preserve_evidence_core=config.get('preserve_evidence_core',False),
                    concise_answers=config.get('concise_answers',False)))
                selected_data=prepared['cf_conversations']
            timings['question_targets_seconds']=time.monotonic()-start
            if prepared['audit'].get('deadline_reached') or time.time()>=deadline:
                timings['total_learning_seconds']=time.monotonic()-started
                write(root/'results/partial'/(key+'.json'),{'case':key,'status':'partial',
                    'reason':'deadline_during_question_preparation','rank':rank,
                    'predictions':predictions,'training':training,'timings':timings,
                    'learning_protocol':protocol,audit_field:prepared['audit'],
                    'benchmark_answers_used_for_training':False})
                del initial,prepared,selected_data,original_train,original_val
                gc.collect();torch.cuda.empty_cache()
                break
            # Both adaptive arms start from exactly the same native initialization.
            query_data=prepared['query_conversations']
            heldout=min(8,max(1,len(guided)//4)) if len(guided)>1 else 0
            heldout_question_ids=set(range(len(guided)-heldout,len(guided))) if heldout else set()
            if config.get('heldout_split','rank_tail')=='seeded_random':
                # Fix membership before admission filtering, independently of answers
                # and reranker order; both adaptive arms use identical IDs.
                heldout_question_ids=set(random.Random(seed).sample(range(len(guided)),heldout))
            for arm,values,filename in [('query_sd',query_data,'query.safetensors'),
                    ('contextual_cartridge',selected_data,'contextual.safetensors')]:
                train=[c for c in values if c.metadata['question_index'] not in heldout_question_ids]
                val=[c for c in values if c.metadata['question_index'] in heldout_question_ids]
                start=time.monotonic();cache,training[arm]=train_arm(engine,initial,train,val,config,seed)
                training[arm]['train_question_ids']=[c.metadata['question_index'] for c in train]
                training[arm]['heldout_question_ids']=[c.metadata['question_index'] for c in val]
                predictions[arm]=engine.generate_student(query,cache,max_new_tokens=config['eval_tokens'])[0]
                cache.clear();timings[arm+'_seconds']=time.monotonic()-start
                if config.get('save_cartridges',True): save_cartridge(cache,root/'cartridges'/key/filename,config['model'])
                del cache;gc.collect();torch.cuda.empty_cache()
            timings['total_learning_seconds']=time.monotonic()-started
            result={'case':key,'task':item['task'],'length':item['length'],'status':'completed',
                'protocol':config.get('protocol',protocol),'learning_protocol':protocol,
                'rank':rank,'predictions':predictions,'training':training,'timings':timings,
                audit_field:prepared['audit'],'dataset':record.get('counts',{}),
                'benchmark_answers_used_for_training':False,'native_kv_only':True,
                'teacher_backbone_frozen':True,'student_backbone_frozen':True,
                'cartridge_inference_uses_source_document':False,
                'source_reading_baselines':['full_context','evidence_teacher'],
                'adaptive_heldout_question_ids':sorted(heldout_question_ids),
                'adaptive_heldout_policy':config.get('heldout_split','rank_tail'),
                'prefix_tokens':config['prefix_tokens'],
                'generation_kernel_options':engine.generation_kernel_options,
                'gpu':torch.cuda.get_device_name()}
            if protocol=='legacy_counterfactual':
                result['proposal_student_state']='native_initialization_before_request_SD'
            write(dest,result);done.append(key)
            print(json.dumps({'event':'learning_case_complete','case':key,'seconds':timings['total_learning_seconds'],
                'learning_protocol':protocol,'selected_conversations':len(selected_data)}),flush=True)
            del initial,prepared,query_data,selected_data,original_train,original_val
            gc.collect();torch.cuda.empty_cache()
        except Exception as error:
            failures.append(key);write(root/'results/failures'/(key+'.json'),
                {'case':key,'rank':rank,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
            # CUDA failures may poison the process; do not mislabel later cases.
            if isinstance(error,torch.OutOfMemoryError) or 'CUDA' in str(error): break
        write(root/'results/workers'/f'{rank:03d}.json',{'rank':rank,'completed':done,'failed':failures,
            'assigned':len(assigned),'timestamp':time.time()})
    write(root/'results/workers'/f'{rank:03d}.json',{'rank':rank,'completed':done,'failed':failures,
        'assigned':len(assigned),'timestamp':time.time(),'terminal':True})
    return 0 if len(done)==len(assigned) else 75 if time.time()>=deadline else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True)
    args=p.parse_args();return run(read(args.config),args.rank)


if __name__=='__main__':raise SystemExit(main())
