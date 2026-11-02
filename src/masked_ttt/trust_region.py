"""Two-direction trust step that incorporates native-prediction preservation.

Both SD KL and native KL have the same categorical output-Fisher quadratic
term; their projected gradients differ. Solve the preservation multiplier
along with the output trust-region multiplier instead of only shrinking an
unconstrained direction. This is a local model, not a correctness oracle.
"""

import math

import torch


def preserving_coordinates(
    gradient,
    curvature,
    native_gradient,
    native_value,
    native_budget,
    *,
    radius=0.15,
    damping=0.05,
    max_coordinate=4.0,
    native_curvature=None,
):
    device = gradient.device
    dtype = gradient.dtype
    b = gradient.detach().double().cpu().tolist()
    h = curvature.detach().double().cpu().tolist()
    c = native_gradient.detach().double().cpu().tolist()
    if len(b) != 2 or len(c) != 2 or len(h) != 2 or any(len(row) != 2 for row in h):
        raise ValueError("This solver requires two complete parameter directions")
    if any(not math.isfinite(x) for x in b + c + [x for row in h for x in row]):
        raise ValueError("Finite local model required")
    if radius <= 0 or damping <= 0 or max_coordinate <= 0 or native_budget < native_value - 1e-9:
        raise ValueError("Positive trust controls and a feasible model origin required")
    h00, h11 = h[0][0], h[1][1]
    h01 = 0.5 * (h[0][1] + h[1][0])
    if min(h00, h11) < -1e-12 or h00 * h11 - h01 * h01 < -1e-10 * max(1.0, h00 * h11):
        raise ValueError("Output curvature must be positive semidefinite")
    h00 = max(h00, 0.0)
    h11 = max(h11, 0.0)
    bound = math.sqrt(h00 * h11)
    h01 = max(-bound, min(bound, h01))
    if native_curvature is None:
        n00, n11, n01 = h00, h11, h01
    else:
        n = native_curvature.detach().double().cpu().tolist()
        if (
            len(n) != 2
            or any(len(row) != 2 for row in n)
            or any(not math.isfinite(x) for row in n for x in row)
        ):
            raise ValueError("Finite two-direction native curvature required")
        n00, n11, n01 = n[0][0], n[1][1], 0.5 * (n[0][1] + n[1][0])
        if min(n00, n11) < -1e-12 or n00 * n11 - n01 * n01 < -1e-10 * max(1.0, n00 * n11):
            raise ValueError("Native curvature must be positive semidefinite")
        n00, n11 = max(n00, 0.0), max(n11, 0.0)
        bound = math.sqrt(n00 * n11)
        n01 = max(-bound, min(bound, n01))
    s0, s1 = math.sqrt(max(h00, 1e-12)), math.sqrt(max(h11, 1e-12))
    a00, a11, a01 = h00 / (s0 * s0), h11 / (s1 * s1), h01 / (s0 * s1)
    na00, na11, na01 = n00 / (s0 * s0), n11 / (s1 * s1), n01 / (s0 * s1)
    bb = [b[0] / s0, b[1] / s1]
    cc = [c[0] / s0, c[1] / s1]

    def quadratic(x):
        return 0.5 * (h00 * x[0] * x[0] + 2 * h01 * x[0] * x[1] + h11 * x[1] * x[1])

    def native_quadratic(x):
        if native_curvature is None:
            return quadratic(x)
        return 0.5 * (n00 * x[0] * x[0] + 2 * n01 * x[0] * x[1] + n11 * x[1] * x[1])

    def preservation(x):
        return native_value + c[0] * x[0] + c[1] * x[1] + native_quadratic(x)

    def objective(x):
        return b[0] * x[0] + b[1] * x[1] + quadratic(x)

    def limit(x):
        factor = min(1.0, max_coordinate / max(max(abs(v) for v in x), 1e-30))
        return [factor * v for v in x]

    def point(lam, mu):
        if native_curvature is None:
            weight = 1.0 + lam + mu
            m00, m11, m01 = weight * a00 + damping, weight * a11 + damping, weight * a01
        else:
            m00 = (1.0 + mu) * a00 + lam * na00 + damping
            m11 = (1.0 + mu) * a11 + lam * na11 + damping
            m01 = (1.0 + mu) * a01 + lam * na01
        r0, r1 = bb[0] + lam * cc[0], bb[1] + lam * cc[1]
        determinant = max(m00 * m11 - m01 * m01, 1e-30)
        return [
            -(m11 * r0 - m01 * r1) / determinant / s0,
            -(m00 * r1 - m01 * r0) / determinant / s1,
        ]

    def trust_point(lam):
        value = point(lam, 0.0)
        if quadratic(value) <= radius:
            return value, 0.0
        upper = 1.0
        while quadratic(point(lam, upper)) > radius and upper < 1e14:
            upper *= 2
        lower = 0.0
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if quadratic(point(lam, middle)) > radius:
                lower = middle
            else:
                upper = middle
        return point(lam, upper), upper

    unconstrained, _ = trust_point(0.0)
    unconstrained = limit(unconstrained)
    # Best feasible point along the original direction is a valid competing
    # candidate. At an active native boundary this can be exactly zero.
    q = quadratic(unconstrained)
    native_q = native_quadratic(unconstrained)
    linear = c[0] * unconstrained[0] + c[1] * unconstrained[1]
    margin = max(0.0, native_budget - native_value)
    if native_q > 1e-30:
        root = (
            2 * margin / (linear + math.sqrt(linear * linear + 4 * native_q * margin))
            if linear > 0
            else (-linear + math.sqrt(linear * linear + 4 * native_q * margin)) / (2 * native_q)
        )
    else:
        root = margin / linear if linear > 0 else 1.0
    upper_fraction = min(1.0, max(0.0, root))
    slope = b[0] * unconstrained[0] + b[1] * unconstrained[1]
    optimal_fraction = (
        min(upper_fraction, max(0.0, -slope / (2 * q))) if q > 1e-30 else upper_fraction
    )
    uniform = [v * optimal_fraction for v in unconstrained]
    lam = mu = 0.0
    candidate = unconstrained
    if preservation(candidate) > native_budget + 1e-12:
        lower = 0.0
        upper = 1.0
        while preservation(trust_point(upper)[0]) > native_budget + 1e-12 and upper < 1e14:
            upper *= 2
        for _ in range(70):
            middle = 0.5 * (lower + upper)
            if preservation(trust_point(middle)[0]) > native_budget:
                lower = middle
            else:
                upper = middle
        lam = upper
        candidate, mu = trust_point(lam)
        candidate = limit(candidate)
    candidates = [
        ([0.0, 0.0], "zero"),
        (uniform, "uniform_scaling"),
        (candidate, "joint_preserving"),
    ]
    valid = [
        (x, name)
        for x, name in candidates
        if preservation(x) <= native_budget + 1e-8 and quadratic(x) <= radius + 1e-8
    ]
    if not valid:
        raise ArithmeticError("The feasible origin was lost numerically")
    chosen, name = min(valid, key=lambda item: objective(item[0]))
    return torch.tensor(chosen, device=device, dtype=dtype), dict(
        selected=name,
        preservation_multiplier=lam,
        trust_multiplier=mu,
        predicted_native_KL=preservation(chosen),
        predicted_teacher_change=objective(chosen),
        predicted_step_KL=quadratic(chosen),
        uniform_scaling_fraction=optimal_fraction,
        uniform_predicted_teacher_change=objective(uniform),
        semantic_correctness_certified=False,
    )
