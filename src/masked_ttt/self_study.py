"""Evidence-grounded self-questions for request-local cartridge training."""

from dataclasses import dataclass
import json
import re
import unicodedata

from .request import Request


@dataclass(frozen=True)
class Exercise:
    question: str
    answer: str
    evidence: str
    source_chunk: int


def normalized(text):
    text=unicodedata.normalize('NFKC',str(text)).casefold()
    return ' '.join(re.findall(r'\w+',text,flags=re.UNICODE))


def answer_is_in_quote(answer,quote):
    """Require a whole textual answer, never a number embedded in another."""
    target=' '.join(unicodedata.normalize('NFKC',answer).casefold().split())
    source=' '.join(unicodedata.normalize('NFKC',quote).casefold().split())
    if not target:return False
    start=0
    while True:
        pos=source.find(target,start)
        if pos<0:return False
        end=pos+len(target)
        left=source[pos-1] if pos else ''
        right=source[end] if end<len(source) else ''
        # ASCII boundaries allow a Chinese unit immediately after a number,
        # but do not accept 320 as part of 1320, 3200, or 320.5.
        left_bad=bool(left and left.isascii() and left.isalnum() and target[0].isascii() and target[0].isalnum())
        right_bad=bool(right and right.isascii() and right.isalnum() and target[-1].isascii() and target[-1].isalnum())
        decimal_bad=(target[-1].isdigit() and right in ('.',',') and end+1<len(source)
                     and source[end+1].isdigit())
        if not (left_bad or right_bad or decimal_bad):return True
        start=pos+1


def parse_exercises(text,document,spans,selected,original_question,limit=2):
    """Parse only exercises anchored verbatim to selected original chunks.

    This structural filter is necessary, not a proof of entailment; generated
    exercises must additionally pass independent question/quote verification.
    """
    if not isinstance(text,str) or limit<1:
        raise ValueError('Need generated text and a positive exercise limit')
    decoder=json.JSONDecoder();items=None
    for i,char in enumerate(text):
        if char not in '[{':continue
        try:candidate,_=decoder.raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if isinstance(candidate,dict):candidate=candidate.get('exercises')
        if isinstance(candidate,list):items=candidate;break
    if items is None:return [],{'invalid_json':1}
    selected_texts=[(i,document[spans[i][0]:spans[i][1]]) for i in selected]
    accepted=[];seen=set();rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for item in items:
        if not isinstance(item,dict) or not all(isinstance(item.get(k),str)
                                                for k in ('question','answer','evidence')):
            reject('invalid_fields');continue
        question,answer,evidence=(item[k].strip() for k in ('question','answer','evidence'))
        key=normalized(question)
        if (not key or len(question)>300 or not answer or len(answer)>100 or
            len(evidence)<4 or len(evidence)>600):
            reject('invalid_lengths');continue
        if key==normalized(original_question) or key in seen:
            reject('duplicate_or_original_question');continue
        chunks=[i for i,chunk in selected_texts if evidence in chunk]
        if not chunks:
            reject('quote_not_in_selected_chunk');continue
        if not answer_is_in_quote(answer,evidence):
            reject('answer_not_in_quote');continue
        accepted.append(Exercise(question,answer,evidence,chunks[0]));seen.add(key)
        if len(accepted)>=limit:break
    return accepted,rejected


def short_answer_equivalent(proposed,verified):
    """Compare independently generated short answers without any gold label."""
    verified=re.sub(r'(?i)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)','',verified.strip())
    def canonical(value):
        words=normalized(value).split()
        if words and words[0] in ('a','an','the'):words=words[1:]
        aliases={'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5'}
        return ' '.join(aliases.get(w,w) for w in words)
    a,b=canonical(proposed),canonical(verified)
    return bool(a and a==b)


def practice_request(original,question,max_new_tokens=16,*,separator='\n\n'):
    """Use the same document prefix with a distinct fact-focused question."""
    start=original.prompt.find(original.document)
    if start<0:raise ValueError('Original source is not in the prompt')
    end=start+len(original.document)
    if not separator or not separator.isspace():
        raise ValueError('Practice separator must be whitespace')
    prompt=original.prompt[:end]+separator+'Practice question: '+question+'\nAnswer concisely using the context.'
    return Request(document=original.document,question=question,prompt=prompt,
                   answer_prefix='',max_new_tokens=max_new_tokens)
