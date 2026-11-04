"""Accept verified short spans without using benchmark answers or new models."""
from contextlib import contextmanager
import re

from masked_ttt.self_study import answer_is_in_quote,normalized,short_answer_equivalent

_BAD=frozenset(('no','not','never','neither','nor','none','nothing','nobody','unknown',
                'unclear','or','maybe','uncertain','cannot'))
_TEMPORAL=frozenset(('first','last','final','previous','before','after','now','current',
                     'recent','latest','earlier','later','original','previously','since',
                     'earliest','initial','afterwards','subsequent','prior','originally','currently'))
_DEICTIC=frozenset(('there','here','somewhere','anywhere','this','that','it','he','she',
                    'they','him','her','them','someone'))
_ALIASES={'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5'}


def _canonical(value):
    value=re.sub(r'(?i)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)','',str(value).strip())
    words=normalized(value).split()
    if words and words[0] in ('a','an','the'):words=words[1:]
    return [_ALIASES.get(word,word) for word in words]


def accepted(exercise,verified):
    """Return (accepted, reason), preserving all original exact matches.

    The proposal has already been selected and quote-anchored by the normal
    parser. The additional path only accepts a unique, independently verified
    short token span of that anchored proposal. Comparative temporal and
    deictic answers are left to the original verifier.
    """
    if not answer_is_in_quote(exercise.answer,exercise.evidence):return False,'not_quote_anchored'
    if short_answer_equivalent(exercise.answer,verified):return True,'original_exact'
    proposal=_canonical(exercise.answer);answer=_canonical(verified)
    question=set(normalized(exercise.question).split())
    if (not proposal or not answer or len(answer)>4 or len(proposal)<=len(answer) or
        set(proposal+answer)&_BAD or question&(_TEMPORAL|_BAD) or set(answer)&_DEICTIC):
        return False,'unsupported_or_ambiguous'
    occurrences=sum(proposal[i:i+len(answer)]==answer for i in range(len(proposal)-len(answer)+1))
    if occurrences!=1:return False,'not_unique_span'
    numbers=[word for word in proposal if word.isdigit()]
    if any(word.isdigit() for word in answer) and len(numbers)!=1:
        return False,'multiple_numbers'
    return True,'verified_anchored_subspan'


@contextmanager
def anchored_short_verifier():
    """Apply the optional verifier to one request and restore runtime hooks.

    It does not supply either generated answer to Student or Teacher prompts.
    Hooks are process-local and cannot be shared with concurrent requests.
    """
    import masked_ttt.self_study_runtime as runtime
    old_quote=runtime.answer_quote;old_compare=runtime.short_answer_equivalent
    pending=[];stats=dict(original_exact=0,verified_anchored_subspan=0,rejected=0)

    def quote(engine,reader,exercise,limit):
        if pending:raise AssertionError('A verification was not consumed')
        result=old_quote(engine,reader,exercise,limit)
        pending.append(exercise)
        return result

    def compare(proposal,verified):
        if len(pending)!=1:raise AssertionError('An independent quote verification is required')
        exercise=pending.pop()
        if exercise.answer!=proposal:raise AssertionError('Verification proposal changed')
        result,reason=accepted(exercise,verified)
        stats[reason if result else 'rejected']+=1
        return result

    runtime.answer_quote=quote;runtime.short_answer_equivalent=compare
    try:yield stats
    finally:
        runtime.answer_quote=old_quote;runtime.short_answer_equivalent=old_compare
