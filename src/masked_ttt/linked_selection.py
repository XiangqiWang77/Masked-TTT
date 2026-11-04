"""Label-free link-aware alternatives at the same Decision-score budget.

The caller supplies document text, the question, source chunk spans and the
already-computed Decisions scores.  Gold answers and support annotations are
neither accepted nor read by the selection functions.
"""
from collections import Counter,defaultdict
import math
import re

from .frozen_scores import FrozenJSONSelector,load_scores
from .selection import PreparedRequest


WORD=re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
SENTENCE=re.compile(r'(?<=[.!?])\s+|\n+')
STOP=frozenset(('about','after','again','along','also','among','and','any','are','as','at',
    'before','been','being','between','but','can','could','did','does','for','from','had',
    'has','have','how','into','its','may','more','most','much','must','not','now','of','on',
    'one','only','or','other','our','out','over','same','she','should','some','such','than',
    'that','the','their','them','then','there','these','they','this','those','through','under',
    'was','were','what','when','where','which','while','who','whom','whose','why','will','with',
    'would','you','your'))


def _words(text):
    return {m.group().lower() for m in WORD.finditer(text) if m.group().lower() not in STOP}


def _names(text):
    return {m.group().lower() for m in WORD.finditer(text)
            if m.group()[0].isupper() and not m.group().isupper() and m.group().lower() not in STOP}


def choose(document,question,spans,records,fraction,policy='score',reserve_fraction=.25):
    n=len(records)
    if n==0 or len(spans)!=n or not 0<fraction<=1 or not 0<=reserve_fraction<1:
        raise ValueError('Invalid selection dimensions or budget')
    if policy not in ('score','neighbor','entity'):
        raise ValueError('Unknown selector')
    probabilities=[]
    for i,(span,record) in enumerate(zip(spans,records)):
        a,b=span;p=record.get('decision_probabilities',{}).get('yes')
        if not 0<=a<b<=len(document) or record.get('chunk')!=i or record.get('utility_output_valid') is not True:
            raise ValueError('Invalid ordered source range or Decision score')
        if type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Invalid Decision probability')
        probabilities.append(p)
    k=max(1,math.ceil(n*fraction))
    ranked=sorted(range(n),key=lambda i:(-probabilities[i],i))
    if policy=='score' or k<3 or reserve_fraction==0:
        return sorted(ranked[:k])
    reserved=min(k-1,max(1,math.ceil(k*reserve_fraction)))
    anchors=ranked[:k-reserved];chosen=set(anchors)
    if policy=='neighbor':
        distance={i:min(abs(i-a) for a in anchors) for i in range(n) if i not in chosen}
        linked=sorted((i for i in distance if distance[i]==1),
                      key=lambda i:(-probabilities[i],i))
    else:
        chunks=[document[a:b] for a,b in spans]
        terms=_words(question)
        links=defaultdict(list)
        for a in anchors:
            for sentence in SENTENCE.split(chunks[a]):
                if _words(sentence)&terms:
                    for name in _names(sentence):links[name].append(a)
        chunk_names=[_names(text) for text in chunks]
        df=Counter(name for names in chunk_names for name in names)
        strength={}
        for i,names in enumerate(chunk_names):
            if i in chosen:continue
            score=0.
            for name in sorted(names&links.keys()):
                closest=min(abs(i-a) for a in links[name])
                anchor=max(probabilities[a] for a in links[name])
                score+=max(anchor,.01)/(math.sqrt(df[name])*(1+closest/4))
            if score>0:strength[i]=score
        linked=sorted(strength,key=lambda i:(-strength[i],-probabilities[i],i))
    for i in linked[:reserved]:chosen.add(i)
    for i in ranked:
        if len(chosen)==k:break
        chosen.add(i)
    if len(chosen)!=k:raise AssertionError('Selection budget changed')
    return sorted(chosen)


class LinkedFrozenSelector(FrozenJSONSelector):
    def prepare(self,backbone,original):
        prepared=super().prepare(backbone,original)
        data=load_scores(self.directory,original.document,original.question,
                         self.case_key,self.by_case)
        mapping=prepared.metadata['source_mapping']
        if mapping is None:
            records=data['records']
        else:
            records=[dict(data['records'][item['original_chunk']],chunk=i)
                     for i,item in enumerate(mapping)]
        selected=choose(prepared.request.document,prepared.request.question,
            prepared.layout.spans,records,backbone.config.keep_fraction,'entity',.25)
        metadata=dict(prepared.metadata)
        metadata.update(selected_chunks=selected,
            selected_original_chunks=[records[i].get('original_chunks',
                [mapping[i]['original_chunk']] if mapping is not None else [i]) for i in selected],
            selected_tokens=sum(len(prepared.layout.chunks[i]) for i in selected),
            selection_policy='decision_entity_link_v1',entity_reserve_fraction=.25,
            reference_answers_used=False,online_api_calls=0)
        return PreparedRequest(prepared.request,prepared.layout,selected,metadata)
