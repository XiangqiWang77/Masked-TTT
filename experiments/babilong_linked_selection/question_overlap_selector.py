"""Label-free question-term reservation at the same Decision chunk budget."""
from collections import Counter
import math
import re

from masked_ttt.linked_selection import WORD, _words, choose

SIMPLE_WHERE_PROPER = re.compile(r"(?i:where\s+is)\s+([A-Z][A-Za-z'-]{2,})\s*\?\s*")


def question_overlap(document, question, spans, records, fraction=.1, *, mode="mix"):
    """Reserve chunks by query-term IDF using only inference-time inputs.

    ``mix`` starts from the established entity-linked selection and replaces
    at most half its reserved slots when an outside candidate has stronger
    query overlap. ``lexical`` reserves the whole 25% for query overlap.
    """
    if mode not in ("mix", "lexical", "proper_where"):
        raise ValueError("Unsupported question-overlap policy")
    baseline = choose(document, question, spans, records, fraction, "entity", .25)
    n, k = len(records), len(baseline)
    if k < 3 or (mode == "proper_where" and not SIMPLE_WHERE_PROPER.fullmatch(question.strip())):
        return baseline
    reserve = min(k-1, max(1, math.ceil(k*.25)))
    probabilities = [record["decision_probabilities"]["yes"] for record in records]
    ranked = sorted(range(n), key=lambda i: (-probabilities[i], i))
    anchors = set(ranked[:k-reserve])
    terms = _words(question)
    counts = [Counter(match.group().lower() for match in WORD.finditer(document[a:b])
                      if match.group().lower() in terms) for a,b in spans]
    frequency = Counter(word for item in counts for word in item)
    weight = {word: math.log((n+1)/(frequency[word]+1)) for word in terms}
    similarity = [sum(min(count, 4)*weight[word] for word,count in item.items()) for item in counts]
    lexical = sorted(range(n), key=lambda i: (-similarity[i], -probabilities[i], i))
    if mode == "lexical":
        chosen = set(anchors)
        for i in lexical:
            if len(chosen) == k:
                break
            if similarity[i] > 0:
                chosen.add(i)
        for i in ranked:
            if len(chosen) == k:
                break
            chosen.add(i)
    else:
        chosen = set(baseline)
        max_replacements = max(1, reserve//2)
        replaced = 0
        for candidate in lexical:
            if replaced >= max_replacements or similarity[candidate] <= 0:
                break
            if candidate in chosen:
                continue
            removable = chosen-anchors
            if not removable:
                break
            weakest = min(removable, key=lambda i: (similarity[i], probabilities[i], -i))
            if similarity[candidate] <= similarity[weakest]:
                continue
            chosen.remove(weakest)
            chosen.add(candidate)
            replaced += 1
    if len(chosen) != k:
        raise AssertionError("The Decision chunk budget changed")
    return sorted(chosen)
