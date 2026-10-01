"""Three-anchor curve: starter 0.25, baseline 0.50, reference 1.00."""

from benchkit.budget import CHECKPOINT_WEIGHTS


def curve(c, S, B, R) -> float:
    """Score of cost `c` against starter S, baseline B and reference R (lower cost is better)."""
    if c is None:
        return 0.0
    # Anchors are measured, so enforce S >= B >= R rather than trust it.
    B = max(B, R)
    S = max(S, B)
    if c <= R:
        return 1.0
    if c <= B:
        return 0.50 + 0.50 * (B - c) / (B - R)
    # Width of the starter band; if starter and baseline tie, taper over the baseline span.
    width = (S - B) or (B - R) or 1
    if c <= S:
        return 0.25 + 0.25 * (S - c) / width
    return 0.25 * max(0.0, 1 - (c - S) / width)


def fill_later(values) -> list:
    """Replace each missing value by its next later one."""
    out = list(values)
    for k in range(len(out) - 2, -1, -1):
        if out[k] is None:
            out[k] = out[k + 1]
    return out


def instance_score(checkpoint_costs, anchor) -> float:
    S = fill_later(anchor["starter_costs"])
    B = fill_later(anchor["baseline_costs"])
    R = anchor["reference_cost"]
    return sum(w * curve(c, S[k], B[k], R)
               for k, (w, c) in enumerate(zip(CHECKPOINT_WEIGHTS, checkpoint_costs)))
