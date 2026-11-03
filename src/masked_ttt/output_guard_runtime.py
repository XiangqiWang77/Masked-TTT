"""Evaluate native versus adapted answers using the frozen full-context model."""

import time
import torch

from .cache import fork_prefix
from .output_guard import decide,same_text
from .request import Request,prepare_layout
from .runtime import private_forward


def label_ids(tokenizer,vocab_size):
    """Collect single-token forms; unsupported tokenizers abstain safely."""
    labels={}
    for name in ('A','B','U'):
        ids=set()
        for literal in (name,' '+name):
            tokens=tokenizer.encode(literal,add_special_tokens=False)
            if len(tokens)==1 and 0<=tokens[0]<vocab_size:ids.add(int(tokens[0]))
        if not ids:raise ValueError('Judge labels are not single-token alternatives')
        labels[name]=ids
    if any(labels[a]&labels[b] for a,b in (('A','B'),('A','U'),('B','U'))):
        raise ValueError('Judge labels map to overlapping token IDs')
    return labels


@torch.no_grad()
def review_order(engine,reader,cache,layout,request,first,second):
    """Score one ordering using the original document and no gold answer."""
    if len(first)>4000 or len(second)>4000:
        raise ValueError('Candidate answer is too long for the comparison prompt')
    instruction=(
        '\n\nCompare these two proposed answers to the original question using the full context. '
        'Check the entities, conditions, dates, negations and any later updates. '
        'Candidate 1: '+first+'\nCandidate 2: '+second+
        '\nReturn only A if Candidate 1 is clearly better supported, B if Candidate 2 '
        'is clearly better supported, or U if they are equally supported or you are unsure. '
        'A and B refer to the candidate numbers, not to multiple-choice options.\nDecision:'
    )
    judge=Request(document=request.document,question=request.question,
        prompt=request.prompt+instruction,answer_prefix='',max_new_tokens=1)
    judge_layout=prepare_layout(engine,judge);length=layout.document_end
    if judge_layout.ids[:length]!=layout.ids[:length]:
        raise ValueError('Judge prompt changed the original document prefix')
    if len(judge_layout.ids)+1>engine.config.max_context:
        raise ValueError('Judge prompt exceeds the model context')
    suffix=torch.tensor(judge_layout.ids[length:],device=engine.device)
    if not len(suffix):raise ValueError('Judge question suffix is empty')
    labels=label_ids(engine.tokenizer,engine.model.config.vocab_size)
    branch=fork_prefix(cache,length);engine.budget.check()
    output=private_forward(reader,suffix,length,branch)
    logits=engine.model.lm_head(output.last_hidden_state[0,-1:])[0].float()
    probabilities=logits.softmax(-1)
    scores={name:float(probabilities[list(ids)].sum()) for name,ids in labels.items()}
    mass=sum(scores.values())
    if not mass>0:return dict(A=0.,B=0.,U=1.,label_mass=0.)
    return {**{name:value/mass for name,value in scores.items()},'label_mass':min(1.,mass)}


def guard_answers(engine,reader,cache,layout,request,base,adapted):
    began=time.perf_counter()
    if same_text(base,adapted):
        decision=decide(base,adapted,None,None,minimum_confidence=engine.config.guard_minimum_confidence,
                        minimum_label_mass=engine.config.guard_minimum_label_mass,
                        policy=engine.config.guard_policy)
        return dict(decision,seconds=0.,reference_answers_used=False)
    try:
        first=review_order(engine,reader,cache,layout,request,base,adapted)
        second=review_order(engine,reader,cache,layout,request,adapted,base)
        decision=decide(base,adapted,first,second,
            minimum_confidence=engine.config.guard_minimum_confidence,
            minimum_label_mass=engine.config.guard_minimum_label_mass,policy=engine.config.guard_policy)
    except ValueError as exc:
        decision=dict(use_adapted=engine.config.guard_policy!='consensus',
                      reason='judge_unavailable',detail=str(exc),reviews=[])
    engine.synchronize()
    return dict(decision,seconds=time.perf_counter()-began,reference_answers_used=False,
                judgment_context='full_working_document',orderings=2)
