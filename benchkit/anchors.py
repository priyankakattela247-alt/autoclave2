"""Anchor generation: median of repeats, spreads recorded and checked.

Real anchors use one worker and the full reference scale. Quick checks may use
several workers and a shorter reference run; their numbers are never written.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from benchkit.aggregate import ANCHOR_SCHEMA
from benchkit.runner import run_instance
from benchkit.scoring import instance_score

REFERENCE_SCALE = 10
QUICK_REFERENCE_SCALE = 2
MIN_SPAN = 0.02          # (B_final - R) / B_final must be at least this ...
MIN_SPAN_UNITS = 8       # ... and B_final - R at least this many cost units (all costs are integers)
MAX_SPREAD = 0.25        # repeat spread must stay under this share of the span
MAX_PROBE_SCORE = 0.80   # release gate: the bench's textbook probe must not score above this


class AnchorError(Exception):
    pass


def min_span(baseline_cost: int) -> int:
    """Smallest acceptable baseline-reference span: max(2% of the baseline, 8 units)."""
    return max(-(-baseline_cost * int(MIN_SPAN * 1000) // 1000), MIN_SPAN_UNITS)


def _median(values):
    ordered = sorted(values, key=lambda v: (v is None, v))  # None counts as worst
    return ordered[len(ordered) // 2]


def _spread(values):
    return None if None in values else max(values) - min(values)


def anchor_instance(bench, entry, repeats=3, log=print, reference_scale=REFERENCE_SCALE) -> tuple:
    """Return (anchor dict, list of problems) for one suite entry."""
    problems = []
    runs = {}
    for role, spec, scale, private in (("starter", bench.starter, 1, False),
                                       ("baseline", bench.baseline, 1, False),
                                       ("reference", bench.reference, reference_scale, True)):
        runs[role] = []
        for i in range(repeats):
            instance = bench.data.make_instance(entry.seed, entry.profile, entry.name)
            rec = run_instance(bench, entry, spec, bench.data.budget_for(instance) * scale,
                               allow_private=private)
            if rec.crashed or rec.overrun or rec.final_cost is None:
                problems.append(f"{role} run {i + 1}: crashed={rec.crashed} overrun={rec.overrun} "
                                f"final={rec.final_cost} {(rec.error or '').strip()[-200:]}")
            runs[role].append(rec)
        log(f"  {entry.name} {role}: finals {[r.final_cost for r in runs[role]]}")

    k_count = len(runs["starter"][0].checkpoint_costs)
    starter_costs = [_median([r.checkpoint_costs[k] for r in runs["starter"]]) for k in range(k_count)]
    baseline_costs = [_median([r.checkpoint_costs[k] for r in runs["baseline"]]) for k in range(k_count)]
    reference_cost = _median([r.final_cost for r in runs["reference"]])
    spreads = {role: _spread([r.final_cost for r in rs]) for role, rs in runs.items()}

    S, B, R = starter_costs[-1], baseline_costs[-1], reference_cost
    if None in (S, B, R):
        problems.append("missing final anchor cost")
    else:
        if not S >= B >= R:
            problems.append(f"anchors out of order: starter {S}, baseline {B}, reference {R}")
        span = B - R
        if span < min_span(B):
            problems.append(f"baseline-reference span {span} is under max({MIN_SPAN:.0%} of {B}, "
                            f"{MIN_SPAN_UNITS} units) = {min_span(B)}")
        for role, width in (("starter", S - B), ("baseline", span), ("reference", span)):
            sp = spreads[role]
            if sp is not None and width > 0 and sp > MAX_SPREAD * width:
                problems.append(f"{role} spread {sp} exceeds {MAX_SPREAD:.0%} of span {width}")

    instance = bench.data.make_instance(entry.seed, entry.profile, entry.name)
    anchor = {
        "digest": instance.digest,
        "family": instance.profile["family"],
        "shifted": instance.profile["shifted"],
        "starter_costs": starter_costs,
        "baseline_costs": baseline_costs,
        "reference_cost": reference_cost,
        "spreads": spreads,
    }

    # Release gate: a one-prompt textbook solver must not come close to the reference.
    if bench.probe and None not in (S, B, R):
        rec = run_instance(bench, entry, bench.probe, bench.data.budget_for(instance),
                           allow_private=True)
        score = instance_score(rec.checkpoint_costs, anchor)
        anchor["probe_score"] = round(score, 4)
        log(f"  {entry.name} probe: final {rec.final_cost} score {score:.3f}")
        if score > MAX_PROBE_SCORE:
            problems.append(f"textbook probe scores {score:.3f} > {MAX_PROBE_SCORE} "
                            f"(probe {rec.final_cost}, baseline {B}, reference {R})")
    return anchor, problems


def generate(bench, entries, repeats=3, log=print, reference_scale=REFERENCE_SCALE,
             workers=1) -> tuple:
    """Anchor every entry. Returns (document, {name: problems}).
    Each worker only waits on child processes, so threads are enough."""
    doc = {"schema": ANCHOR_SCHEMA, "generator_schema": bench.data.SCHEMA_VERSION,
           "repeats": repeats, "reference_scale": reference_scale, "budgets": {}, "instances": {}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda e: anchor_instance(bench, e, repeats, log, reference_scale), entries))
    problems = {}
    for entry, (anchor, issues) in zip(entries, results):
        instance = bench.data.make_instance(entry.seed, entry.profile, entry.name)
        doc["budgets"][entry.name] = bench.data.budget_for(instance)
        doc["instances"][entry.name] = anchor
        if issues:
            problems[entry.name] = issues
    return doc, problems


def write(doc, path) -> None:
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
