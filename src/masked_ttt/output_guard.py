"""Order-consistent full-context comparison of native and adapted answers."""

import math


def same_text(a,b):
    """Only skip comparison for answers equal apart from whitespace."""
    return ' '.join(a.split())==' '.join(b.split())


def decide(base,adapted,review_ab,review_ba,*,minimum_confidence,minimum_label_mass):
    """Accept only when both answer orders favor the adapted answer."""
    if same_text(base,adapted):
        return dict(use_adapted=True,reason='same_answer',reviews=[])
    reviews=[review_ab,review_ba]
    if any(not isinstance(x,dict) for x in reviews):
        return dict(use_adapted=False,reason='judge_unavailable',reviews=[])
    for review in reviews:
        if not all(isinstance(review.get(k),(float,int)) and math.isfinite(review[k])
                   for k in ('A','B','U','label_mass')):
            return dict(use_adapted=False,reason='invalid_judge_scores',reviews=reviews)
        if (any(not 0<=review[k]<=1 for k in ('A','B','U','label_mass')) or
            not math.isclose(sum(review[k] for k in ('A','B','U')),1.,abs_tol=1e-4)):
            return dict(use_adapted=False,reason='invalid_judge_scores',reviews=reviews)
    # First order: A=base, B=adapted. Second order: A=adapted, B=base.
    accepted=(review_ab['label_mass']>=minimum_label_mass and
              review_ba['label_mass']>=minimum_label_mass and
              review_ab['B']>=minimum_confidence and review_ba['A']>=minimum_confidence)
    return dict(use_adapted=accepted,reason='adapted_preferred_in_both_orders' if accepted
                else 'keep_full_context_baseline',reviews=reviews)
