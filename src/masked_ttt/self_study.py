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
    decoder=json.JSONDecoder();items=None;recovered=False
    for i,char in enumerate(text):
        if char not in '[{':continue
        try:candidate,_=decoder.raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if isinstance(candidate,dict):candidate=candidate.get('exercises')
        if isinstance(candidate,list):items=candidate;break
    if items is None:
        # A token limit can truncate the outer array after a complete item.
        # Recover only fully decoded entries within the declared exercises
        # array; normal source and independent-answer checks still apply.
        match=re.search(r'"exercises"\s*:\s*\[',text)
        if match:
            partial=[];pos=match.end()
            while pos<len(text):
                while pos<len(text) and text[pos].isspace():pos+=1
                if partial and pos<len(text) and text[pos]==',':
                    pos+=1
                    while pos<len(text) and text[pos].isspace():pos+=1
                if pos>=len(text) or text[pos]!='{':break
                try:candidate,end=decoder.raw_decode(text,pos)
                except json.JSONDecodeError:break
                if not isinstance(candidate,dict):break
                partial.append(candidate);pos=end
            if partial:items=partial;recovered=True
    if items is None:return [],{'invalid_json':1}
    selected_texts=[(i,document[spans[i][0]:spans[i][1]]) for i in selected]
    accepted=[];seen=set();rejected={'partial_json_recovered':1} if recovered else {}
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
        aliases={'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
                 'years':'year','months':'month','days':'day'}
        return ' '.join(aliases.get(w,w) for w in words)
    a,b=canonical(proposed),canonical(verified)
    return bool(a and a==b)


def verified_answer_hint(proposed,verified,evidence):
    """Return a quote-anchored short answer corroborated by a second reader.

    The generator sometimes returns the whole supporting sentence as its
    answer.  In that case, accept the verifier's shorter extract only if it
    occurs in both that sentence and the selected quote.  Ambiguous numeric,
    disjunctive, negative, and uncertain answers are not shortened.
    """
    verification=normalized(re.sub(r'(?i)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)',
                                   '',verified.strip()))
    if verification in {'unknown','unsure','not known','not specified','not mentioned','none','no answer'}:
        return None
    if short_answer_equivalent(proposed,verified):
        return proposed.strip() if answer_is_in_quote(proposed,evidence) else None
    if re.search(r'(?i)\b(?:not|never|neither|nor|unknown|uncertain|either|or)\b',proposed):
        return None
    text=re.sub(r'(?i)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)','',verified.strip())
    text=text.strip(' \t\r\n.,;:!?"\'')
    text=re.sub(r'(?i)^(?:a|an|the)\s+','',text)
    candidates=[text]
    location=re.search(r'(?i)\b(?:is|are|was|were)\s+(?:in|at|on|inside)\s+(?:the\s+)?([^,;.!?]+)$',text)
    if location:candidates.append(location.group(1).strip())
    numbers=set(re.findall(r'(?<![\w.])\d+(?:[.,]\d+)?',proposed))
    for candidate in candidates:
        key=normalized(candidate)
        if (not key or len(key.split())>5 or key in
            {'unknown','unsure','not known','not specified','not mentioned','none','no answer'}):
            continue
        if any(ch.isdigit() for ch in candidate) and len(numbers)>1:
            continue
        if answer_is_in_quote(candidate,proposed) and answer_is_in_quote(candidate,evidence):
            return candidate
    return None


def practice_request(original,question,max_new_tokens=16):
    """Use the same document prefix with a distinct fact-focused question."""
    start=original.prompt.find(original.document)
    if start<0:raise ValueError('Original source is not in the prompt')
    end=start+len(original.document)
    prompt=original.prompt[:end]+'\n\nPractice question: '+question+'\nAnswer concisely using the context.'
    return Request(document=original.document,question=question,prompt=prompt,
                   answer_prefix='',max_new_tokens=max_new_tokens)


def teacher_practice_request(student,evidence_answer):
    """Add the independently checked hint only to the teacher's suffix."""
    if not evidence_answer or student.document in evidence_answer:
        raise ValueError('Invalid evidence answer hint')
    prompt=(student.prompt+'\nFor this practice only, the source-supported short answer is: '
            +evidence_answer+'\nGive the answer concisely.')
    return Request(document=student.document,question=student.question,prompt=prompt,
                   answer_prefix='',max_new_tokens=student.max_new_tokens)
