"""Conservative preservation of a shared query-condition prefix.

This uses the given question and the existing native/teacher predictions;
it never uses reference answers or a new teacher. A qualifier already copied
in their shared prefix makes a subsequent confident query-word substitution
a reason to preserve the native prefix during memory updates.
"""

import re

import torch


STOP_WORDS = set(
    "a an the is are was were am be been being do does did what where when which who whom whose why how please tell me in on at of to for with from by and or as".split()
)
QUALIFIERS = set(
    "before after prior subsequent earlier later initial original final latest earliest first last approved planned actual maximum minimum greater less more fewer above below under over left right north south east west not without except".split()
)


def question_token_sets(tokenizer, question):
    content = set()
    qualifiers = set()
    for word in re.findall(r"\w+", question or ""):
        normalized = word.casefold()
        if normalized in STOP_WORDS:
            continue
        for variant in {word, word.lower(), word.capitalize()}:
            for text in (variant, " " + variant):
                ids = tokenizer.encode(text, add_special_tokens=False)
                # Whole-word variants avoid treating common subword pieces
                # as trusted entities or qualifiers.
                if len(ids) == 1:
                    content.add(ids[0])
                    if normalized in QUALIFIERS:
                        qualifiers.add(ids[0])
    return tuple(sorted(content)), tuple(sorted(qualifiers))


@torch.no_grad()
def query_prefix_guard(
    native_probabilities,
    teacher_probabilities,
    content_ids,
    qualifier_ids,
    *,
    minimum_margin=0.5,
    budget=0.25,
    diagnostics=None,
):
    diagnostics = {} if diagnostics is None else diagnostics

    def reject(reason):
        diagnostics.update(eligible=False, reason=reason)
        return None

    if not content_ids or not qualifier_ids:
        return reject("No complete query qualifier/content tokens")
    native_values, native_ids = native_probabilities.topk(2, dim=-1)
    teacher_ids = teacher_probabilities.argmax(-1)
    differences = (native_ids[:, 0] != teacher_ids).nonzero().flatten()
    if not len(differences):
        return reject("Native and teacher greedy prefixes agree")
    first = int(differences[0])
    diagnostics.update(
        first_mismatch=first,
        native_id=int(native_ids[first, 0]),
        teacher_id=int(teacher_ids[first]),
        native_top1_probability=float(native_values[first, 0]),
        native_top2_probability=float(native_values[first, 1]),
    )
    # A different initial answer style is insufficient evidence of a
    # changed condition. Require an actual shared qualifier-bearing prefix.
    if first == 0:
        return reject("Initial answer style differs; no shared qualifier-bearing prefix")
    content = set(content_ids)
    qualifiers = set(qualifier_ids)
    diagnostics.update(
        native_token_in_question=int(native_ids[first, 0]) in content,
        teacher_token_in_question=int(teacher_ids[first]) in content,
    )
    if int(native_ids[first, 0]) not in content:
        return reject("Native differing token is not a question word")
    if int(teacher_ids[first]) in content:
        return reject("Teacher differing token is also a question word")
    if not qualifiers.intersection(teacher_ids[:first].tolist()):
        return reject("No shared query qualifier")
    margins = native_values[:, 0] - native_values[:, 1]
    if float(margins[first]) < minimum_margin:
        return reject("Native condition margin below the configured threshold")
    diagnostics.update(
        eligible=True, reason="Confident native query condition replaced by a non-question token"
    )
    positions = torch.arange(first + 1, device=native_ids.device)
    budgets = (0.45 * margins[: first + 1].square()).clamp(min=1e-7, max=budget)
    return dict(
        positions=positions,
        native_ids=native_ids[: first + 1, 0],
        budgets=budgets,
        critical_position=first,
        native_condition_id=int(native_ids[first, 0]),
        teacher_condition_id=int(teacher_ids[first]),
        native_condition_margin=float(margins[first]),
        interpretation="Preserves a confident query-linked prefix decision; not a general factual-correctness certificate.",
    )
