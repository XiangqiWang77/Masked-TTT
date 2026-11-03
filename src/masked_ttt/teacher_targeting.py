"""Find a multiple-choice position in the teacher's own answer, never gold."""
import re


def choice_positions(tokenizer,token_ids):
    text=tokenizer.decode(token_ids,skip_special_tokens=True)
    patterns=[r'(?i)\b(?:the\s+correct\s+answer\s+is|the\s+answer\s+is|final\s+answer\s*(?:is|:)|answer\s*:)\s*\(?([ABCD])\)?(?=[\s.,;:!?]|$)',
              r'(?im)^\s*\(?([ABCD])\)?\s*[.!]?\s*$']
    matches=[]
    for pattern in patterns:
        matches.extend((m.start(),m.end(1)) for m in re.finditer(pattern,text))
    if not matches:return []
    _,end=max(matches)
    desired=text[:end]
    for stop in range(1,len(token_ids)+1):
        if tokenizer.decode(token_ids[:stop],skip_special_tokens=True).startswith(desired):
            return [stop-1]
    return []
