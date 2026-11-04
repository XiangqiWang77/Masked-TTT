"""One predeclared alternative to greedy-only Student prefixes.

This is an isolated single-request experiment. The teacher still scores each
actual Student prefix; the loss, selected chunks and KV-only update are intact.
"""
from contextlib import contextmanager, nullcontext

import torch

from experiments.prompt_alignment.same_gpu_pair import _run_pair
from masked_ttt.runtime import private_forward


@torch.no_grad()
def sampled_rollout(engine,reader,cache,suffix,limit,generator):
    """Draw tokens from the current Student's full-vocabulary distribution."""
    if suffix.ndim!=1 or not len(suffix) or limit<1:
        raise ValueError("A nonempty question and positive rollout budget are required")
    branch=reader.compiled_branch(cache)
    engine.budget.check()
    output=private_forward(reader.reader,suffix,reader.memory.document_length,branch)
    tokens=[]
    while len(tokens)<limit:
        logits=engine.model.lm_head(output.last_hidden_state[0,-1:])[0].float()
        probabilities=logits.softmax(-1)
        if not bool(torch.isfinite(probabilities).all()) or float(probabilities.sum())<=0:
            raise FloatingPointError("Nonfinite Student sampling probabilities")
        token=int(torch.multinomial(probabilities,1,generator=generator))
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        output=private_forward(reader.reader,torch.tensor([token],device=engine.device),
            reader.memory.document_length+len(suffix)+len(tokens)-1,branch)
    return torch.tensor(tokens,dtype=torch.long,device=engine.device)


@contextmanager
def mixed_rollouts(*,seed):
    """Use a greedy first update and seeded on-policy samples on alternating calls.

    A private generator avoids changing global RNG. This diagnostic patches a
    process-local function and must not run concurrent requests in that process.
    """
    if type(seed) is not int or seed<0:
        raise ValueError("A nonnegative integer sampling seed is required")
    import masked_ttt.pipeline as pipeline
    original=pipeline.student_rollout
    generators={};stats=dict(calls=0,greedy_calls=0,sampled_calls=0,seed=seed)

    def alternate(engine,reader,cache,suffix,limit):
        index=stats["calls"];stats["calls"]+=1
        if index%2==0:
            stats["greedy_calls"]+=1
            return original(engine,reader,cache,suffix,limit)
        device=torch.device(engine.device)
        key=str(device)
        if key not in generators:
            generators[key]=torch.Generator(device=device).manual_seed(seed)
        stats["sampled_calls"]+=1
        return sampled_rollout(engine,reader,cache,suffix,limit,generators[key])

    pipeline.student_rollout=alternate
    try:yield stats
    finally:pipeline.student_rollout=original


def run_mixed_pair(engine,selector,request,config,*,time_budget_seconds=None):
    """Pair original greedy SD with a mixed-prefix candidate on the same model."""
    state={}
    @contextmanager
    def candidate():
        with mixed_rollouts(seed=config.seed) as observed:
            try:yield
            finally:state.update(observed)
    result=_run_pair(engine,selector,request,config,config,
        contexts=(nullcontext,candidate),require_same_practices=True,
        time_budget_seconds=time_budget_seconds)
    result.update(changed_settings=["student_prefix_sampling"],
                  sampling=dict(state,temperature=1.,first_update="greedy",
                                teacher_scores_student_prefix=True,reference_answers_used=False))
    return result
