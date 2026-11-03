"""Generate and independently check selected-evidence practice questions."""

from types import SimpleNamespace
import re
import time

import torch
from transformers.cache_utils import DynamicCache

from ._layout import selected_read_plan
from .cache import fork_prefix
from .request import Request,prepare_layout
from .runtime import private_forward
from .self_study import (parse_exercises,practice_request,teacher_practice_request,
                         verified_answer_hint)


def suffixes(engine,original_layout,selected,teacher,new_request):
    """Build new question suffixes while keeping both document caches exact."""
    layout=prepare_layout(engine,new_request);length=original_layout.document_end
    if layout.ids[:length]!=original_layout.ids[:length]:
        raise ValueError('Practice prompt changed the original document token prefix')
    plan=selected_read_plan(layout.ids,original_layout.chunks,original_layout.initial,selected,
                            engine.tokenizer.encode('\n\n',add_special_tokens=False))
    if tuple(plan.ids[:plan.document_end])!=teacher.document_ids:
        raise ValueError('Practice prompt changed the selected Teacher document prefix')
    student=torch.tensor(layout.ids[length:],device=engine.device)
    focused=SimpleNamespace(document_cache=teacher.document_cache,document_end=teacher.document_end,
        question_tokens=torch.tensor(plan.ids[plan.document_end:],device=engine.device))
    if not len(student) or not len(focused.question_tokens):
        raise ValueError('Practice question suffix must be nonempty')
    return student,focused


@torch.no_grad()
def generate_from_cache(engine,reader,cache,length,suffix,limit):
    """Greedy generation from a fresh branch; never writes source KV."""
    if suffix.ndim!=1 or not len(suffix) or limit<1:raise ValueError('Invalid generation suffix')
    branch=fork_prefix(cache,length)
    engine.budget.check()
    output=private_forward(reader,suffix,length,branch)
    tokens=[]
    while len(tokens)<limit:
        token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        output=private_forward(reader,torch.tensor([token],device=engine.device),
            length+len(suffix)+len(tokens)-1,branch)
    return tokens,engine.tokenizer.decode(tokens,skip_special_tokens=True)


@torch.no_grad()
def answer_quote(engine,reader,exercise,limit):
    """Independently answer the small question using only its quoted evidence."""
    prompt=('Use only the evidence below. Answer the question with the shortest '
            'supported answer. If the evidence does not answer the question, say unknown.\n\n'
            'Evidence: '+exercise.evidence+'\nQuestion: '+exercise.question+'\nAnswer:')
    text=engine.tokenizer.apply_chat_template([dict(role='user',content=prompt)],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=torch.tensor(engine.tokenizer.encode(text,add_special_tokens=False),device=engine.device)
    if not len(ids) or len(ids)+limit>engine.config.max_context:
        raise ValueError('Evidence verification prompt exceeds the model context')
    cache=DynamicCache(config=engine.model.config)
    engine.budget.check();output=private_forward(reader,ids,0,cache)
    tokens=[]
    while len(tokens)<limit:
        token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        output=private_forward(reader,torch.tensor([token],device=engine.device),len(ids)+len(tokens)-1,cache)
    return engine.tokenizer.decode(tokens,skip_special_tokens=True)


def generate_self_study(engine,teacher_reader,teacher,request,layout,selected):
    """Return verified practice paths; all content originates from selected text."""
    began=time.perf_counter()
    if not engine.config.self_question_count:
        return [],dict(generated_tokens=0,parsed=0,verified=0,checks=[],rejections={},
                       seconds=0.,reference_answers_used=False,student_answer_hint_used=False)
    instruction=(
        '\n\nFor self-study, propose at most '+str(engine.config.self_question_count)+
        ' short factual practice questions needed for the original question. '
        'Each must ask for one explicitly stated fact in one context passage. '
        'Return only JSON: {"exercises":[{"question":"...","answer":"...",'
        '"evidence":"an exact quote supporting that answer"}]}. '
        'The answer must be the shortest verbatim answer span in the quote, '
        'not the full sentence. Do not solve the original question.'
    )
    generation_request=Request(document=request.document,question=request.question,
        prompt=request.prompt+instruction,answer_prefix='',max_new_tokens=engine.config.self_question_max_tokens)
    _,focused=suffixes(engine,layout,selected,teacher,generation_request)
    generated,text=generate_from_cache(engine,teacher_reader,focused.document_cache,
        focused.document_end,focused.question_tokens,engine.config.self_question_max_tokens)
    proposals,rejections=parse_exercises(text,request.document,layout.spans,selected,
        request.question,engine.config.self_question_count)
    verified=[];checks=[];boundary_recovered=0
    for item in proposals:
        answer=answer_quote(engine,teacher_reader,item,engine.config.self_question_verify_tokens)
        hint=verified_answer_hint(item.answer,answer,item.evidence)
        accepted=hint is not None
        checks.append(dict(question=item.question,answer=item.answer,evidence=item.evidence,
                           source_chunk=item.source_chunk,verifier_answer=answer,
                           verified_answer_hint=hint,accepted=accepted))
        if not accepted:continue
        question=practice_request(request,item.question,engine.config.student_rollout_tokens)
        try:
            student,_=suffixes(engine,layout,selected,teacher,question)
            teacher_question=teacher_practice_request(question,hint)
            _,focused=suffixes(engine,layout,selected,teacher,teacher_question)
        except ValueError as exc:
            # Some tokenizers merge the last document character with the first
            # newline. Retry only this exact mismatch using the original
            # prompt's whitespace, then repeat all prefix/cache checks.
            end=request.prompt.find(request.document)+len(request.document)
            match=re.match(r'^\s+',request.prompt[end:])
            separator=match.group() if match else ''
            if (engine.config.practice_boundary_fallback and
                str(exc)=='Practice prompt changed the original document token prefix' and
                separator and separator!='\n\n'):
                try:
                    question=practice_request(request,item.question,engine.config.student_rollout_tokens,
                                              separator=separator)
                    student,_=suffixes(engine,layout,selected,teacher,question)
                    teacher_question=teacher_practice_request(question,hint)
                    _,focused=suffixes(engine,layout,selected,teacher,teacher_question)
                except ValueError as retry_error:
                    checks[-1]['accepted']=False;checks[-1]['reason']='document_prefix_mismatch'
                    checks[-1]['boundary_fallback_error']=str(retry_error);continue
                checks[-1]['boundary_fallback_used']=True;boundary_recovered+=1
            else:
                checks[-1]['accepted']=False;checks[-1]['reason']='document_prefix_mismatch';continue
        verified.append(dict(name='practice_'+str(len(verified)+1),student_suffix=student,
                             teacher=focused,question=item.question))
    engine.synchronize()
    return verified,dict(generated_tokens=len(generated),parsed=len(proposals),verified=len(verified),
        checks=checks,rejections=rejections,generated_preview=text[:512],seconds=time.perf_counter()-began,
        reference_answers_used=False,student_answer_hint_used=False,teacher_answer_hint_used=bool(verified),
        boundary_fallback_recovered=boundary_recovered)
