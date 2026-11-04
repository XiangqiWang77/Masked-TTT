"""A label-free two-hop entity bridge under the same Decision-score budget."""
from collections import Counter,defaultdict
import math

from masked_ttt.linked_selection import SENTENCE,_names,_words,choose


def two_hop(document,question,spans,records,fraction=.1,*,reserve_fraction=.25,
            require_score_improvement=False):
    """Replace at most half the linked slots with two-hop entity candidates.

    The score-ranked anchors are never removed. All candidates and links are
    derived from the document, the question and cached Decision probabilities;
    this function accepts no support labels or answer annotations.
    """
    if type(require_score_improvement) is not bool:
        raise ValueError("Score protection must be boolean")
    baseline=choose(document,question,spans,records,fraction,"entity",reserve_fraction)
    n,k=len(records),len(baseline)
    if k<3 or reserve_fraction==0:return baseline
    reserve=min(k-1,max(1,math.ceil(k*reserve_fraction)))
    probabilities=[record["decision_probabilities"]["yes"] for record in records]
    ranked=sorted(range(n),key=lambda i:(-probabilities[i],i))
    anchors=set(ranked[:k-reserve]);chunks=[document[a:b] for a,b in spans]
    names=[_names(text) for text in chunks]
    frequency=Counter(name for group in names for name in group)
    terms=_words(question);question_names=_names(question)
    origins=defaultdict(set)
    for index in anchors:
        for sentence in SENTENCE.split(chunks[index]):
            if _words(sentence)&terms or _names(sentence)&question_names:
                for name in _names(sentence):origins[name].add(index)
    if not origins:return baseline
    bridges=defaultdict(list)
    for index in baseline:
        text=chunks[index]
        for sentence in SENTENCE.split(text):
            found=_names(sentence)
            for seed in found&origins.keys():
                closest=min(abs(index-a) for a in origins[seed])
                anchor=max(probabilities[a] for a in origins[seed])
                strength=max(anchor,.01)*math.sqrt(max(probabilities[index],.001))/(
                    math.sqrt(frequency[seed])*(1+closest/4))
                for other in found-origins.keys():
                    bridges[other].append((index,strength))
    if not bridges:return baseline
    chosen=set(baseline);strengths={};requirements={}
    for index,found in enumerate(names):
        if index in chosen or probabilities[index]<.001:continue
        options=[]
        for name in sorted(found&bridges.keys()):
            for source,weight in bridges[name]:
                score=weight/((1+abs(index-source)/4)*math.sqrt(frequency[name]))
                options.append((score,-source))
        if options:
            score,negative_source=max(options)
            if score>0:strengths[index]=score;requirements[index]=-negative_source
    candidates=sorted(strengths,key=lambda i:(-strengths[i],-probabilities[i],i))
    maximum=max(1,reserve//2)
    protected=set();replaced=0
    for candidate in candidates:
        if replaced>=maximum:break
        source=requirements[candidate]
        if source not in chosen:continue
        removable=chosen-anchors-protected-{source}
        if not removable:continue
        weakest=min(removable,key=lambda i:(probabilities[i],-i))
        if require_score_improvement and probabilities[candidate]<=probabilities[weakest]:continue
        chosen.remove(weakest);chosen.add(candidate)
        protected.update((source,candidate));replaced+=1
    if len(chosen)!=k or not anchors.issubset(chosen):
        raise AssertionError("The Decision budget or anchors changed")
    return sorted(chosen)
