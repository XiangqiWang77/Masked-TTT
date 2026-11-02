"""Label-free admission and budgets for Selective SD.

Unresolved evidence is a separate state, not a negative correctness label.
The uncertain budget is calibrated globally; no benchmark answer enters here.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SelectiveSDPolicy:
    supported_steps: int = 16
    uncertain_steps: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.supported_steps, self.uncertain_steps)):
            raise ValueError("Step budgets must be nonnegative integers")
        if self.uncertain_steps > self.supported_steps:
            raise ValueError("Uncertain signals cannot receive a larger budget")


def assessment_state(assessment):
    """Distinguish valid uncertainty from failed or contradictory parsing."""
    if not isinstance(assessment, dict) or assessment.get("parse_valid") is not True:
        return "invalid"
    fields = ("candidate_has_answer", "candidate_supported", "candidate_contradicted")
    if any(type(assessment.get(key)) is not bool for key in fields):
        return "invalid"
    has_answer, supported, contradicted = (assessment[key] for key in fields)
    grounded = assessment.get("source_quotes_verified") is True
    if supported and contradicted:
        return "invalid"
    if not has_answer:
        return "no_answer"
    if contradicted:
        return "contradicted" if grounded else "invalid"
    if supported:
        return "supported" if grounded else "invalid"
    # An explicit, valid "neither established" verdict is usable uncertainty.
    if assessment.get("resolved") is False and assessment.get("apply_update") is None:
        return "unresolved"
    return "invalid"


def decide_update(assessment, policy):
    state = assessment_state(assessment)
    steps = (
        policy.supported_steps
        if state == "supported"
        else policy.uncertain_steps
        if state == "unresolved"
        else 0
    )
    return dict(
        state=state,
        planned_steps=steps,
        apply_update=bool(steps),
        policy=asdict(policy),
        reference_answers_used=False,
    )
