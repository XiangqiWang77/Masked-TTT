"""Function-space checked, joint g/DeltaV updates with a fixed teacher.

Only the update rule changes. Curvature is the two-dimensional output Fisher
in the current g and V descent directions, not a claim that the teacher is
factually correct. Native-prefix KL is an optional preservation constraint;
it is neither a new teacher nor a correctness certificate.
"""

from dataclasses import asdict, dataclass
import math

import torch

from .trust_region import preserving_coordinates
from .query_guard import query_prefix_guard


@dataclass(frozen=True)
class UpdateOptions:
    output_radius: float = 0.15
    # A fixed cumulative anchor can make a confident native error impossible
    # to correct. Keep it available as an explicit ablation, never a default
    # certificate of teacher reliability. Per-step trust checks still apply.
    native_KL_budget: float | None = None
    gain_probe: float = 0.25
    value_probe: float = 0.02
    damping: float = 0.05
    derivative_tolerance: float = 0.5
    minimum_ratio: float = 0.1
    max_coordinate: float = 4.0
    backtracks: int = 7
    diagonal_curvature: bool = False
    geometry: str = "output_fisher"
    freeze_initial_gradient: bool = False
    allow_native_zero_boundary: bool = False
    direction_preconditioner: str = "raw_gradient"
    adaptive_probes: bool = False
    preservation_aware_geometry: bool = False
    adam_gradient_clip: float = 10.0
    query_guard: bool = False
    query_guard_budget: float = 0.25
    query_guard_minimum_margin: float = 0.5

    def __post_init__(self):
        for name in (
            "output_radius",
            "gain_probe",
            "value_probe",
            "damping",
            "derivative_tolerance",
            "minimum_ratio",
            "max_coordinate",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Positive finite " + name + " required")
        if self.native_KL_budget is not None and (
            not math.isfinite(self.native_KL_budget) or self.native_KL_budget < 0
        ):
            raise ValueError("Nonnegative native KL budget or None required")
        if type(self.backtracks) is not int or self.backtracks < 1:
            raise ValueError("At least one backtracking trial required")
        if self.geometry not in ("output_fisher", "first_order"):
            raise ValueError("Declare output_fisher or first_order geometry")
        if self.freeze_initial_gradient and self.geometry != "first_order":
            raise ValueError("Frozen-gradient ablation must use first-order geometry")
        if self.direction_preconditioner not in ("raw_gradient", "adam"):
            raise ValueError("Declare raw_gradient or adam direction preconditioning")
        if self.preservation_aware_geometry and self.geometry != "output_fisher":
            raise ValueError("Preservation-aware geometry requires an output-Fisher model")
        if self.adam_gradient_clip <= 0:
            raise ValueError("Positive Adam gradient clip required")
        if not 0 < self.query_guard_budget or not math.isfinite(self.query_guard_budget):
            raise ValueError("Positive finite query guard budget required")
        if not 0 <= self.query_guard_minimum_margin <= 1:
            raise ValueError("Query guard margin must be a probability difference")
        if self.query_guard and not self.preservation_aware_geometry:
            raise ValueError("Query guard requires preservation-aware geometry")


def position_kl(logits, probabilities):
    logp = logits.log_softmax(-1)
    return (torch.special.xlogy(probabilities, probabilities) - probabilities * logp).sum(-1)


def distribution_kl(logits, probabilities):
    return position_kl(logits, probabilities).mean()


def output_fisher(logits, derivatives):
    """Exact categorical Fisher projected onto arbitrary logit directions."""
    probabilities = logits.double().softmax(-1)
    directions = torch.stack(derivatives).double()
    centered = directions - (directions * probabilities).sum(-1, keepdim=True)
    return torch.einsum("atv,btv,tv->ab", centered, centered, probabilities) / len(logits)


def trust_coordinates(gradient, curvature, options):
    matrix = curvature.double()
    if options.diagonal_curvature:
        matrix = torch.diag(matrix.diag())
    scales = matrix.diag().clamp_min(1e-12).sqrt()
    whitened = matrix / (scales[:, None] * scales[None, :])
    identity = torch.eye(len(gradient), device=matrix.device, dtype=matrix.dtype)
    step = (
        -torch.linalg.solve(whitened + options.damping * identity, gradient.double() / scales)
        / scales
    )
    magnitude = float(step.abs().max())
    if magnitude > options.max_coordinate:
        step *= options.max_coordinate / magnitude
    cost = float(0.5 * step @ matrix @ step)
    if cost > options.output_radius:
        step *= math.sqrt(options.output_radius / cost)
    if not bool(torch.isfinite(step).all()) or float(gradient.double() @ step) >= 0:
        step = -gradient.double() / gradient.double().abs().max().clamp_min(1e-12)
        cost = float(0.5 * step @ matrix @ step)
        if cost > options.output_radius:
            step *= math.sqrt(options.output_radius / cost)
    return step


def normalized_directions(memory, gradients, options):
    g, v = gradients
    # g has one scalar per layer/head/chunk; V has one low-rank matrix per
    # layer/head/chunk. Set probe sizes without letting parameter counts
    # determine the relative weight of these two different units.
    dg = -g / g.abs().max().clamp_min(1e-20) * options.gain_probe
    dv = -v / v.flatten(-2).norm(dim=-1).max().clamp_min(1e-20) * options.value_probe
    dg = dg * memory.selected_chunk_mask[None, None, :]
    return dg, dv


@torch.no_grad()
def proposed_directions(memory, gradients, options, state):
    if options.direction_preconditioner == "raw_gradient":
        return normalized_directions(memory, gradients, options), None
    count = 1 if state is None else state["step"] + 1
    norm = sum(float(g.square().sum()) for g in gradients) ** 0.5
    scale = min(1.0, options.adam_gradient_clip / max(norm, 1e-30))
    old_m = [torch.zeros_like(g) for g in gradients] if state is None else state["m"]
    old_v = [torch.zeros_like(g) for g in gradients] if state is None else state["v"]
    moments = [m * 0.9 + g * (0.1 * scale) for m, g in zip(old_m, gradients)]
    variances = [v * 0.999 + g.square() * (0.001 * scale * scale) for v, g in zip(old_v, gradients)]
    preconditioned = [
        (m / (1 - 0.9**count)) / ((v / (1 - 0.999**count)).sqrt() + 1e-8)
        for m, v in zip(moments, variances)
    ]
    directions = normalized_directions(memory, preconditioned, options)
    if state is not None and sum(float((g * d).sum()) for g, d in zip(gradients, directions)) >= 0:
        directions, restarted = proposed_directions(memory, gradients, options, None)
        restarted["restarted"] = True
        return directions, restarted
    return directions, dict(step=count, m=moments, v=variances, restarted=False)


def fit_memory(
    memory,
    evaluate,
    probabilities,
    steps,
    options,
    check=lambda: None,
    *,
    query_token_ids=(),
    query_qualifier_ids=(),
):
    """Optimize fixed KL, accepting only verified function-space steps.

    Grad and no-grad calls share the causal incremental operator. The
    native/augmented zero boundary is measured explicitly; its offset
    enters the local prediction and every proposal still faces actual-loss
    and native-preservation checks. No benchmark labels enter this interface.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("Nonnegative integer steps required")
    params = (memory.gain, memory.value_coefficients)
    memory.value_enabled = True
    for p in params:
        p.requires_grad_(True)
    with torch.no_grad():
        native = evaluate().detach()
        native_probabilities = native.softmax(-1)
        current = native.clone()
        guard_observation = {}
        guard = (
            query_prefix_guard(
                native_probabilities,
                probabilities,
                query_token_ids,
                query_qualifier_ids,
                minimum_margin=options.query_guard_minimum_margin,
                budget=options.query_guard_budget,
                diagnostics=guard_observation,
            )
            if options.query_guard
            else None
        )
    guard_proof = dict(
        enabled=options.query_guard, triggered=guard is not None, observation=guard_observation
    )
    if guard is not None:
        guard_proof.update({k: (v.tolist() if torch.is_tensor(v) else v) for k, v in guard.items()})

    def query_status(logits):
        if guard is None:
            return dict(passed=True, triggered=False)
        divergences = position_kl(logits, native_probabilities)[guard["positions"]]
        argmax_retained = bool((logits.argmax(-1)[guard["positions"]] == guard["native_ids"]).all())
        bounded = bool((divergences <= guard["budgets"] + 1e-7).all())
        return dict(
            passed=argmax_retained and bounded,
            triggered=True,
            prefix_argmax_retained=argmax_retained,
            position_KL=divergences.tolist(),
        )

    # The caller must start from the exact zero-adapter native state.
    if any(bool(p.detach().any()) for p in params):
        raise ValueError("Native-preservation reference requires zero initialization")
    initial = float(distribution_kl(current, probabilities))
    records = []
    frozen = None
    initial_gradients = None
    initial_tangent_origin = None
    direction_state = None
    forwards = 1
    backwards = 0
    radius = options.output_radius
    for index in range(steps):
        check()
        for p in params:
            p.grad = None
        if options.freeze_initial_gradient and frozen is not None:
            gradients = frozen
            gap = None
            model_current = current
            zero_boundary = False
        else:
            logits = evaluate()
            forwards += 1
            loss = distribution_kl(logits, probabilities)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite teacher KL")
            gap = float((logits.detach() - current).abs().max())
            zero_boundary = (
                options.allow_native_zero_boundary
                and gap != 0
                and not any(bool(p.detach().any()) for p in params)
            )
            if gap != 0 and not zero_boundary:
                raise AssertionError(
                    "Nonzero-state gradient/evaluation logit mismatch: " + str(gap)
                )
            model_current = logits.detach() if gap else current
            loss.backward()
            backwards += 1
            gradients = tuple(p.grad.detach().clone() for p in params)
            del logits, loss
            if options.freeze_initial_gradient:
                frozen = gradients
        before = float(distribution_kl(current, probabilities))
        model_before = float(distribution_kl(model_current, probabilities))
        boundary_offset = before - model_before
        if initial_tangent_origin is None:
            initial_tangent_origin = model_before
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Nonfinite memory derivative")
        if bool(gradients[0][:, :, ~memory.selected_chunk_mask].any()):
            raise AssertionError("Unselected g received a gradient")
        if initial_gradients is None:
            initial_gradients = gradients
        gradient_alignment = []
        for old, new in zip(initial_gradients, gradients):
            denominator = float(old.norm() * new.norm())
            gradient_alignment.append(
                float((old * new).sum()) / denominator if denominator else None
            )
        directions, proposed_state = proposed_directions(
            memory, gradients, options, direction_state
        )
        projected = torch.stack([(g * d).sum().double() for g, d in zip(gradients, directions)])
        snapshot = memory.snapshot()
        for p in params:
            p.grad = None

        def propose(coordinates):
            memory.restore(snapshot)
            for p, d, a in zip(params, directions, coordinates):
                p.add_(d, alpha=float(a))
            memory.constrain()

        with torch.no_grad():
            errors = []
            slope_error = None
            reliable = False
            probe_details = []
            solver_proof = None
            matrix = torch.zeros(2, 2, device=current.device, dtype=torch.double)
            if options.geometry == "output_fisher":
                derivatives = []
                for group in range(2):
                    cache_samples = {}
                    choices = []
                    best = None
                    best_score = float("inf")
                    scales = (1.0, 0.5, 2.0, 0.25, 4.0) if options.adaptive_probes else (1.0,)
                    for scale in scales:
                        estimates = []
                        for width in (scale, scale * 0.5):
                            pair = []
                            for sign in (1.0, -1.0):
                                location = sign * width
                                if location not in cache_samples:
                                    coords = [0.0, 0.0]
                                    coords[group] = location
                                    propose(coords)
                                    cache_samples[location] = evaluate()
                                    forwards += 1
                                pair.append(cache_samples[location])
                            estimates.append((pair[0] - pair[1]) / (2 * width))
                        coarse, fine = estimates
                        weights = model_current.softmax(-1)

                        def centered(x):
                            return x - (x * weights).sum(-1, keepdim=True)

                        coarse, fine = centered(coarse), centered(fine)
                        error = float(
                            ((coarse - fine).square() * weights).sum().sqrt()
                            / (fine.square() * weights).sum().sqrt().clamp_min(1e-12)
                        )
                        fd_slope = float((fine * (weights - probabilities)).sum() / len(current))
                        floor = max(float(projected.abs().sum()) * 0.05, 1e-12)
                        ad_error = abs(fd_slope - float(projected[group])) / max(
                            abs(float(projected[group])), floor
                        )
                        score = max(error, ad_error) if options.adaptive_probes else error
                        choices.append(
                            dict(scale=scale, multiscale_error=error, autograd_slope_error=ad_error)
                        )
                        if score < best_score:
                            best_score = score
                            best = (fine, error, scale, ad_error)
                        if not options.adaptive_probes or score <= options.derivative_tolerance:
                            break
                    fine, error, scale, ad_error = best
                    derivatives.append(fine)
                    errors.append(error)
                    probe_details.append(
                        dict(selected_scale=scale, autograd_slope_error=ad_error, trials=choices)
                    )
                memory.restore(snapshot)
                matrix = output_fisher(model_current, derivatives)
                observed_matrix = matrix
                if options.diagonal_curvature:
                    matrix = torch.diag(matrix.diag())
                fd = torch.stack(
                    [
                        (d * (model_current.softmax(-1) - probabilities)).sum().double()
                        / len(current)
                        for d in derivatives
                    ]
                )
                slope_error = float((fd - projected).norm() / projected.norm().clamp_min(1e-12))
                checks = errors + [slope_error]
                if options.adaptive_probes:
                    checks.extend(p["autograd_slope_error"] for p in probe_details)
                reliable = max(checks) <= options.derivative_tolerance
                local = UpdateOptions(**dict(asdict(options), output_radius=radius))
                coordinates = trust_coordinates(projected, matrix, local)
                if (
                    reliable
                    and options.preservation_aware_geometry
                    and options.native_KL_budget is not None
                ):
                    native_gradient = torch.stack(
                        [
                            (d * (model_current.softmax(-1) - native_probabilities)).sum().double()
                            / len(current)
                            for d in derivatives
                        ]
                    )
                    native_value = max(
                        0.0, float(distribution_kl(model_current, native_probabilities))
                    )
                    if native_value <= options.native_KL_budget + 1e-7:
                        coordinates, solver_proof = preserving_coordinates(
                            projected,
                            matrix,
                            native_gradient,
                            native_value,
                            options.native_KL_budget + 1e-7,
                            radius=radius,
                            damping=options.damping,
                            max_coordinate=options.max_coordinate,
                        )
                    else:
                        solver_proof = dict(
                            selected="original_zero_boundary_fallback", model_origin_feasible=False
                        )
                if reliable and guard is not None:
                    # The mean KL can hide a large change to one condition.
                    # If needed, rotate the same two directions subject to
                    # that position's own Fisher and native distribution.
                    guard_models = []
                    for position, budget in zip(
                        guard["positions"].tolist(), guard["budgets"].tolist()
                    ):
                        per_derivatives = [d[position : position + 1] for d in derivatives]
                        per_matrix = output_fisher(
                            model_current[position : position + 1], per_derivatives
                        )
                        per_gradient = torch.stack(
                            [
                                (
                                    d[position]
                                    * (
                                        model_current[position].softmax(-1)
                                        - native_probabilities[position]
                                    )
                                )
                                .sum()
                                .double()
                                for d in derivatives
                            ]
                        )
                        per_value = max(
                            0.0,
                            float(
                                position_kl(
                                    model_current[position : position + 1],
                                    native_probabilities[position : position + 1],
                                )[0]
                            ),
                        )
                        predicted = per_value + float(
                            per_gradient @ coordinates
                            + 0.5 * coordinates @ per_matrix @ coordinates
                        )
                        guard_models.append(
                            (
                                predicted - budget,
                                position,
                                budget,
                                per_value,
                                per_gradient,
                                per_matrix,
                            )
                        )
                    worst = max(guard_models, key=lambda x: x[0])
                    if worst[0] > 0:
                        _, position, budget, per_value, per_gradient, per_matrix = worst
                        if per_value <= budget + 1e-7:
                            coordinates, position_proof = preserving_coordinates(
                                projected,
                                matrix,
                                per_gradient,
                                per_value,
                                budget + 1e-7,
                                radius=radius,
                                damping=options.damping,
                                max_coordinate=options.max_coordinate,
                                native_curvature=per_matrix,
                            )
                            solver_proof = dict(
                                mean_native_solver=solver_proof,
                                query_position=position,
                                query_position_solver=position_proof,
                            )
                        else:
                            solver_proof = dict(
                                mean_native_solver=solver_proof,
                                query_position=position,
                                query_model_origin_feasible=False,
                            )
            else:
                coordinates = torch.ones_like(projected) * min(
                    1.0, math.sqrt(radius / options.output_radius)
                )
            if options.geometry == "output_fisher" and not reliable:
                # A failed numerical curvature check is never presented as
                # second-order evidence. Use a small descent proposal, still
                # subject to the same actual KL and retention constraints.
                coordinates = torch.ones_like(projected) * min(
                    0.25, math.sqrt(radius / options.output_radius) * 0.25
                )
            accepted = False
            trials = []
            after = before
            chosen = None
            for trial in range(options.backtracks):
                fraction = 0.5**trial
                actual = coordinates * fraction
                predicted = boundary_offset + float(
                    -projected @ actual - 0.5 * actual @ matrix @ actual
                )
                propose(actual)
                candidate = evaluate()
                forwards += 1
                value = float(distribution_kl(candidate, probabilities))
                retention = float(distribution_kl(candidate, native_probabilities))
                move = float(distribution_kl(candidate, current.softmax(-1)))
                reduction = before - value
                ratio = reduction / max(predicted, 1e-12)
                finite = all(math.isfinite(x) for x in (value, retention, move))
                preserve = (
                    options.native_KL_budget is None or retention <= options.native_KL_budget + 1e-7
                )
                query_check = query_status(candidate)
                prediction_verified = reliable and predicted > 0 and ratio >= options.minimum_ratio
                accept = (
                    finite
                    and reduction > max(1e-9, abs(before) * 1e-7)
                    and preserve
                    and query_check["passed"]
                    and move <= radius * 1.1 + 1e-7
                    and (not reliable or prediction_verified or zero_boundary and predicted <= 0)
                )
                trials.append(
                    dict(
                        fraction=fraction,
                        teacher_KL=value,
                        native_KL=retention,
                        step_KL=move,
                        predicted_reduction=predicted,
                        actual_reduction=reduction,
                        actual_predicted_ratio=ratio,
                        prediction_verified=prediction_verified,
                        query_preservation=query_check,
                        accepted=accept,
                    )
                )
                if accept:
                    accepted = True
                    after = value
                    current = candidate
                    chosen = trials[-1]
                    break
            if not accepted:
                memory.restore(snapshot)
                radius = max(radius * 0.5, 1e-5)
            elif reliable and chosen["actual_predicted_ratio"] > 0.75:
                radius = min(options.output_radius, radius * 1.5)
            if accepted:
                direction_state = proposed_state
            first_order_prediction = initial_tangent_origin + sum(
                float((g * p.detach()).sum()) for g, p in zip(initial_gradients, params)
            )
            coupling = float(matrix[0, 1] / (matrix[0, 0] * matrix[1, 1]).clamp_min(1e-24).sqrt())
            records.append(
                dict(
                    step=index,
                    joint_values=True,
                    accepted=accepted,
                    teacher_KL_before=before,
                    teacher_KL_after=after,
                    teacher_KL_gradient_path=(
                        initial_tangent_origin
                        if options.freeze_initial_gradient and index
                        else model_before
                    ),
                    local_model_teacher_KL_intercept=model_before,
                    gradient_evaluation_logit_gap=gap,
                    native_zero_operator_boundary=zero_boundary,
                    native_to_gradient_teacher_KL_offset=boundary_offset,
                    g_gradient_norm=float(gradients[0].norm()),
                    V_gradient_norm=float(gradients[1].norm()),
                    unselected_g_gradient_zero=True,
                    source_targets_backpropagated=0,
                    source_target_spans=[],
                    source_chunks=[],
                    source_NTP=0.0,
                    optimizer_state_scope="joint_function_space_trust_region",
                    direction_preconditioner=options.direction_preconditioner,
                    committed_preconditioner_steps=direction_state["step"]
                    if direction_state is not None
                    else 0,
                    preconditioner_state_committed=accepted and proposed_state is not None,
                    preconditioner_restarted=bool(
                        proposed_state and proposed_state.get("restarted")
                    ),
                    geometry=options.geometry,
                    gradient_refreshed=not options.freeze_initial_gradient or index == 0,
                    gradient_cosine_to_initial_by_group=gradient_alignment,
                    initial_tangent_teacher_KL_prediction=first_order_prediction,
                    initial_tangent_prediction_error=after - first_order_prediction,
                    output_Fisher_group_correlation=coupling
                    if options.geometry == "output_fisher"
                    else None,
                    observed_output_Fisher_group_correlation=(
                        float(
                            observed_matrix[0, 1]
                            / (observed_matrix[0, 0] * observed_matrix[1, 1])
                            .clamp_min(1e-24)
                            .sqrt()
                        )
                        if options.geometry == "output_fisher"
                        else None
                    ),
                    projected_gradient=projected.tolist(),
                    output_Fisher=matrix.tolist(),
                    derivative_multiscale_relative_errors=errors,
                    finite_difference_probes=probe_details,
                    preservation_solver=solver_proof,
                    derivative_error_norm="categorical_output_Fisher",
                    derivative_autograd_relative_error=slope_error,
                    curvature_verified=reliable,
                    proposed_coordinates=coordinates.tolist(),
                    trials=trials,
                    native_KL_after=float(distribution_kl(current, native_probabilities)),
                    native_max_position_KL_after=float(
                        position_kl(current, native_probabilities).max()
                    ),
                    native_top1_agreement=float(
                        (current.argmax(-1) == native.argmax(-1)).float().mean()
                    ),
                    query_preservation=query_status(current),
                    cumulative_forward_evaluations=forwards,
                    cumulative_backward_evaluations=backwards,
                    next_radius=radius,
                )
            )
    with torch.no_grad():
        final = float(distribution_kl(current, probabilities))
        retained = dict(
            native_prefix_KL=float(distribution_kl(current, native_probabilities)),
            native_KL_budget=options.native_KL_budget,
            native_KL_is_acceptance_constraint=options.native_KL_budget is not None,
            step_output_KL_radius=options.output_radius,
            native_prefix_top1_agreement=float(
                (current.argmax(-1) == native.argmax(-1)).float().mean()
            ),
            native_prefix_KL_by_position=position_kl(current, native_probabilities).tolist(),
            native_prefix_max_position_KL=float(position_kl(current, native_probabilities).max()),
            original_teacher_KL=initial,
            original_document_KV_unchanged=True,
            initial_tangent_origin_teacher_KL=initial_tangent_origin,
            native_zero_boundary_observed=any(
                r["native_zero_operator_boundary"] and r["gradient_evaluation_logit_gap"]
                for r in records
            ),
            forward_evaluations=forwards,
            backward_evaluations=backwards,
            constraint_scope="same teacher answer prefixes, not every possible query",
            query_preservation=dict(guard_proof, final=query_status(current)),
            semantic_correctness_certified=False,
            teacher_probabilities_changed=False,
        )
    return records, final, retained
