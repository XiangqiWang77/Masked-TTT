"""Compare cartridges trained on full documents that differ only outside E.

The manually fixed chunks isolate a causal question; they are not a replacement
for the Decision API selector used in the benchmark. Candidate answer strings
are supplied only to the post-training diagnostic, never to the training loss.
"""
from contextlib import contextmanager
from dataclasses import replace
import math
import re

import torch
from torch.nn import functional as F

from masked_ttt import MaskedTTT,Request
from masked_ttt.cache import fork_prefix
from masked_ttt.frozen_scores import layout_on_source_spans
from masked_ttt.memory_accounting import module_tensors
from masked_ttt.reader import matched_incremental_logits
from masked_ttt.runtime import private_forward
from masked_ttt.selection import PreparedRequest
from masked_ttt.standalone import build_standalone


def make_pair():
    """One intervention changes a location in an unselected middle chunk."""
    def build(location):
        chunks=['Daniel took the milk.\n']
        chunks+=['A neutral note was logged.\n']*8
        chunks+=['Daniel moved to the '+location+'.\n']
        chunks+=['Another neutral note.\n']*9
        chunks+=['Daniel dropped the milk.\n']
        spans=[];at=0
        for text in chunks:
            spans.append((at,at+len(text)));at+=len(text)
        return ''.join(chunks),spans
    a,spans_a=build('garden');b,spans_b=build('office')
    if len(spans_a)!=20 or [a[s:e] for s,e in (spans_a[0],spans_a[-1])] != [
        b[s:e] for s,e in (spans_b[0],spans_b[-1])]:
        raise AssertionError('The selected evidence changed')
    if a[:spans_a[9][0]]!=b[:spans_b[9][0]] or a[spans_a[9][1]:]!=b[spans_b[9][1]:]:
        raise AssertionError('The intervention changed more than one unselected chunk')
    return [(a,spans_a,'garden'),(b,spans_b,'office')]


class FixedSelector:
    def __init__(self,spans):self.spans=spans

    def prepare(self,backbone,request):
        layout=layout_on_source_spans(backbone,request,self.spans)
        return PreparedRequest(request,layout,[0,len(self.spans)-1],dict(
            backend='controlled_fixed_selection',reference_answers_used=False,
            online_api_calls=0,complete_original_document=True,
            selected_chunks=[0,len(self.spans)-1],total_chunks=len(self.spans)))


@torch.no_grad()
def _score(engine,reader,source,suffix,target):
    tokens=torch.tensor(engine.tokenizer.encode(' '+target,add_special_tokens=False),
                        dtype=torch.long,device=engine.device)
    if not len(tokens):raise ValueError('Candidate needs at least one token')
    logits=matched_incremental_logits(engine,reader,source,suffix,tokens)
    logp=F.log_softmax(logits.float(),dim=-1).gather(-1,tokens[:,None]).sum()
    value=float(logp)
    if not math.isfinite(value):raise ValueError('Non-finite candidate likelihood')
    return dict(log_probability=value,token_count=len(tokens))


@torch.no_grad()
def _score_physical(cartridge,target):
    """Score a candidate using only released, materialized cartridge KV.

    The target tokens are fed after training and after constructing a fresh
    question suffix branch. They never enter the adaptation loss or prompt.
    """
    cartridge._assert_intact()
    engine=cartridge._engine
    targets=torch.tensor(engine.tokenizer.encode(' '+target,add_special_tokens=False),
                         dtype=torch.long,device=engine.device)
    if not len(targets):raise ValueError('Candidate needs at least one token')
    branch=fork_prefix(cartridge._branch,cartridge._length)
    engine.budget.check()
    output=private_forward(cartridge._reader.reader,cartridge._suffix,cartridge._length,branch)
    values=[]
    for index,token in enumerate(targets):
        logits=engine.model.lm_head(output.last_hidden_state[0,-1:]).float()
        values.append(F.log_softmax(logits,dim=-1)[0,token])
        if index+1<len(targets):
            engine.budget.check()
            output=private_forward(cartridge._reader.reader,targets[index:index+1],
                cartridge._length+len(cartridge._suffix)+index,branch)
    value=float(torch.stack(values).sum())
    if not math.isfinite(value):raise ValueError('Non-finite physical candidate likelihood')
    cartridge._assert_intact()
    return dict(log_probability=value,token_count=len(targets))


@torch.no_grad()
def _decode(engine,reader,source,suffix,length,limit):
    branch=reader.compiled_branch(source)
    engine.budget.check();out=private_forward(reader.reader,suffix,length,branch)
    tokens=[]
    while len(tokens)<limit:
        token=int(engine.model.lm_head(out.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        out=private_forward(reader.reader,torch.tensor([token],device=engine.device),
                            length+len(suffix)+len(tokens)-1,branch)
    return dict(prediction=engine.tokenizer.decode(tokens,skip_special_tokens=True),tokens=tokens)


@contextmanager
def trace_selected_cache(candidates=('garden','office'),full_limit=8):
    """Measure selected-cache output after training and before final decode."""
    import masked_ttt.pipeline as pipeline
    from masked_ttt.self_study_runtime import generate_from_cache
    original=pipeline.answer_standalone;original_prefill=pipeline.prefill_document
    records=[];full_baselines=[]

    @torch.no_grad()
    def prefill(engine,reader,ids,initial,boundaries):
        cache=original_prefill(engine,reader,ids,initial,boundaries)
        if full_baselines:raise AssertionError('Expected one full-document prefill')
        stamps=[(x,x._version,x.data_ptr()) for layer in cache.layers
                for x in (layer.keys,layer.values) if x is not None]
        suffix=torch.tensor(ids[boundaries[-1]:],device=engine.device)
        tokens,prediction=generate_from_cache(engine,reader,cache,boundaries[-1],suffix,full_limit)
        if any(t._version!=version or t.data_ptr()!=pointer for t,version,pointer in stamps):
            raise AssertionError('Full baseline mutated the document KV')
        full_baselines.append(dict(tokens=tokens,prediction=prediction))
        return cache

    @torch.no_grad()
    def trace(engine,layout,selected,teacher,trained,limit):
        source=teacher.document_cache
        stamps=[(x,x._version,x.data_ptr()) for layer in source.layers
                for x in (layer.keys,layer.values) if x is not None]
        reader,branch,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
        del branch
        memory=reader.memory;memory.select_actual_read_path()
        snapshot=memory.snapshot()
        learned={candidate:_score(engine,reader,source,suffix,candidate) for candidate in candidates}
        learned_output=_decode(engine,reader,source,suffix,length,limit)
        try:
            memory.gain.zero_();memory.value_coefficients.zero_();memory.select_actual_read_path()
            if not memory.native_mode:raise AssertionError('Zero control is not native')
            zero={candidate:_score(engine,reader,source,suffix,candidate) for candidate in candidates}
            zero_output=_decode(engine,reader,source,suffix,length,limit)
        finally:memory.restore(snapshot)
        if any(t._version!=version or t.data_ptr()!=pointer for t,version,pointer in stamps):
            raise AssertionError('The selected source KV was mutated')
        records.append(dict(selected_document_ids=tuple(teacher.document_ids),
            suffix_ids=tuple(int(i) for i in suffix.tolist()),
            teacher_suffix_ids=tuple(int(i) for i in teacher.question_tokens.tolist()),
            learned=learned,zero=zero,learned_output=learned_output,zero_output=zero_output,
            gain=trained.gain.detach().float().cpu().clone(),
            value_coefficients=trained.value_coefficients.detach().float().cpu().clone(),
            value_basis=trained.value_basis.detach().float().cpu().clone()))
        del reader,memory,source,stamps,snapshot
        return original(engine,layout,selected,teacher,trained,limit)

    pipeline.answer_standalone=trace;pipeline.prefill_document=prefill
    try:yield records,full_baselines
    finally:
        pipeline.answer_standalone=original;pipeline.prefill_document=original_prefill


def _difference(left,right):
    if left.shape!=right.shape:raise ValueError('States do not have matching shapes')
    return float(torch.linalg.vector_norm(left-right))


def _mentions_only(text,wanted,other):
    return (bool(re.search(r'\b'+re.escape(wanted)+r'\b',text,flags=re.I)) and
            not bool(re.search(r'\b'+re.escape(other)+r'\b',text,flags=re.I)) and
            not bool(re.search(r"\b(?:not|unknown|cannot|can't|unsure|uncertain)\b",text,flags=re.I)))


def run_pair(engine,*,steps=2,max_new_tokens=8,rollout_tokens=3,time_budget_seconds=300):
    """Return factual comparisons without making an effectiveness guarantee."""
    old=engine.config
    try:
        engine.config=replace(old,keep_fraction=.1,steps=steps,max_answer_steps=steps,
            self_question_count=0,student_rollout_tokens=rollout_tokens,guard_enabled=False,
            selected_transfer_weight=0.,value_basis_source='document',
            standalone_cartridge_enabled=True,release_full_before_cartridge=True,
            release_selected_source_before_decode=True)
        answers=[];all_traces=[]
        for document,spans,correct in make_pair():
            request=Request(document=document,question='Where is the milk?',max_new_tokens=max_new_tokens)
            cartridge=None
            try:
                with trace_selected_cache(full_limit=max_new_tokens) as (traces,full):
                    cartridge,result=MaskedTTT(engine,FixedSelector(spans)).prepare_cartridge(
                        request,time_budget_seconds=time_budget_seconds)
                if len(traces)!=1 or len(full)!=1:raise AssertionError('Expected one full and one standalone decode')
                standalone=result['standalone_cartridge'];physical=standalone['memory']
                if (not result['backbone_unchanged'] or not result['original_document_KV_unchanged'] or
                    result['training']['selected_transfer_weight']!=0 or
                    result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']!=0 or
                    not physical['kv_only_materialized'] or physical['cartridge_adapter_bytes']!=0 or
                    physical['additional_source_kv_bytes_before_decode']!=0 or
                    physical['additional_training_adapter_bytes_before_decode']!=0 or
                    module_tensors(cartridge._reader.memory) or
                    standalone['generated_token_ids']!=traces[0]['learned_output']['tokens'] or
                    cartridge.answer(time_budget_seconds=time_budget_seconds)['generated_token_ids']!=standalone['generated_token_ids'] or
                    result['generated_token_ids']!=standalone['generated_token_ids']):
                    raise AssertionError('The physical standalone cartridge contract changed')
                physical_scores={candidate:_score_physical(cartridge,candidate) for candidate in ('garden','office')}
                differences=[abs(physical_scores[c]['log_probability']-traces[0]['learned'][c]['log_probability'])
                             for c in ('garden','office')]
                if any(not math.isclose(physical_scores[c]['log_probability'],traces[0]['learned'][c]['log_probability'],
                                        rel_tol=.005,abs_tol=.02) for c in ('garden','office')):
                    raise AssertionError('Dynamic and physically materialized candidate scores differ')
                traces[0]['physical']=physical_scores
                traces[0]['dynamic_physical_max_abs_logprob_difference']=max(differences)
                answers.append(dict(correct=correct,whole=full[0]['prediction'],
                    teacher=result['teacher']['prediction'],teacher_tokens=result['teacher']['tokens'],
                    zero=traces[0]['zero_output']['prediction'],
                    cartridge=standalone['prediction'],training_steps=len(result['training']['steps'])))
                all_traces.append(traces[0])
            finally:
                if cartridge is not None:cartridge.close()
        x,y=all_traces
        same_selected=(x['selected_document_ids']==y['selected_document_ids'] and
                       x['suffix_ids']==y['suffix_ids'] and x['teacher_suffix_ids']==y['teacher_suffix_ids'])
        if not same_selected:raise AssertionError('The selected teacher input or question changed')
        if answers[0]['teacher_tokens']!=answers[1]['teacher_tokens']:
            raise AssertionError('The fixed selected teacher generated different answers')
        zero_delta=max(abs(x['zero'][candidate]['log_probability']-y['zero'][candidate]['log_probability'])
                       for candidate in ('garden','office'))
        if zero_delta>1e-4 or x['zero_output']['tokens']!=y['zero_output']['tokens']:
            raise AssertionError('The unmodified selected-cache control changed')
        def margin(record,state):
            return record[state]['garden']['log_probability']-record[state]['office']['log_probability']
        m_a=margin(x,'physical');m_b=margin(y,'physical');baseline=margin(x,'zero')
        pair_correct=all(_mentions_only(a['cartridge'],a['correct'],
            'office' if a['correct']=='garden' else 'garden') for a in answers)
        whole_correct=all(_mentions_only(a['whole'],a['correct'],
            'office' if a['correct']=='garden' else 'garden') for a in answers)
        return dict(controlled_question='Where is the milk?',
            selected_evidence=['Daniel took the milk.','Daniel dropped the milk.'],
            intervention='Change only the unselected location between garden and office.',
            selected_inputs_identical=True,zero_cache_max_logprob_difference=zero_delta,
            candidates_used_after_training_only=True,teacher_answer_in_final_prompt=False,
            whole_cache_released_before_cartridge=True,source_cache_released_before_scoring=True,
            training_adapter_released_before_scoring=True,probabilities_from_materialized_kv=True,
            dynamic_physical_max_abs_logprob_difference=max(
                x['dynamic_physical_max_abs_logprob_difference'],y['dynamic_physical_max_abs_logprob_difference']),
            answers=answers,
            candidate_token_counts={k:x['zero'][k]['token_count'] for k in ('garden','office')},
            zero_margin_garden_minus_office=baseline,
            cartridge_margin_garden_context=m_a,cartridge_margin_office_context=m_b,
            margin_difference_garden_minus_office_context=m_a-m_b,
            state_difference_norms=dict(gain=_difference(x['gain'],y['gain']),
                value_coefficients=_difference(x['value_coefficients'],y['value_coefficients']),
                fixed_value_basis=_difference(x['value_basis'],y['value_basis'])),
            full_baseline_mentions_expected_for_both=whole_correct,
            cartridge_mentions_expected_for_both=pair_correct,
            interpretation='A signed margin difference measures sensitivity to the unselected '
                'intervention, not a general answer-quality claim.')
    finally:engine.config=old
