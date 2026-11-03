"""Ground a focused teacher's proposed answer before adapting a request."""

import json


def _quote_in_source(quote,source):
    return bool(isinstance(quote,str) and quote.strip() and
                ' '.join(quote.split()) in ' '.join(source.split()))


def assess(text,source,*,policy,candidate_complete=True,candidate_has_text=True):
    """Interpret a judge's own assessment; do not use evaluation answers.

    Quote containment verifies provenance only.  It does not prove the
    judge's claimed relation or contradiction is logically valid.
    """
    if policy not in ('support_required','grounded_veto'):
        raise ValueError('Unknown teacher-judgment policy')
    fallback=bool(policy=='grounded_veto' and candidate_complete and candidate_has_text)
    result=dict(apply_update=fallback,parse_valid=False,quotes_verified=False,
                contradiction=False,reason='unresolved_judgment',evidence=[])
    if not candidate_complete or not candidate_has_text:
        return dict(result,apply_update=False,reason='incomplete_teacher_answer')
    if not isinstance(text,str):return result
    value=text.strip()
    if value.startswith('```'):
        lines=value.splitlines()
        if len(lines)>=3 and lines[-1].strip()=='```':value='\n'.join(lines[1:-1]).strip()
    try:
        parsed=json.loads(value)
    except (ValueError,TypeError):return result
    if not isinstance(parsed,dict):return result
    reason,evidence=parsed.get('reason'),parsed.get('evidence')
    if (not isinstance(reason,str) or not reason.strip() or not isinstance(evidence,list) or
        any(not isinstance(q,str) or not q.strip() for q in evidence)):
        return result
    has_answer=parsed.get('candidate_has_answer')
    contradicted=parsed.get('candidate_contradicted')
    supported=parsed.get('candidate_supported')
    if policy=='grounded_veto':
        if type(has_answer) is not bool or type(contradicted) is not bool:return result
    elif type(supported) is not bool:
        return result
    grounded=bool(evidence) and all(_quote_in_source(q,source) for q in evidence)
    veto=bool(grounded and contradicted) if policy=='grounded_veto' else False
    accepted=(bool(has_answer) and not veto) if policy=='grounded_veto' else bool(supported and grounded)
    return dict(apply_update=accepted,parse_valid=True,quotes_verified=grounded,
                contradiction=veto,reason=reason,evidence=evidence,
                declared_has_answer=has_answer,declared_contradicted=contradicted,
                declared_supported=supported)


def verified_focused_quotes(judgment,selected_source,full_source):
    """Pass literal evidence only; the first critic's verdict stays private."""
    if not judgment.get('parse_valid'):return []
    return [q for q in judgment.get('evidence',[])
            if _quote_in_source(q,selected_source) and _quote_in_source(q,full_source)]
