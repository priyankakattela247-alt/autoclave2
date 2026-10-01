"""Console tables and JSON reports."""

from dataclasses import asdict


def _cell(bench, cost):
    return "-" if cost is None else bench.format_cost(cost)


def print_table(bench, records, anchors=None, points=None, out=print) -> None:
    rows = [("instance", "size", "family", "shift", "final", "starter", "baseline",
             "reference", "score")]
    for r in records:
        a = (anchors or {}).get("instances", {}).get(r.name) if anchors else None
        status = "overrun" if r.overrun else "crash" if r.crashed else None
        rows.append((
            r.name, str(r.size), r.family, "yes" if r.shifted else "no",
            status if status and r.final_cost is None else _cell(bench, r.final_cost),
            _cell(bench, a["starter_costs"][-1]) if a else "-",
            _cell(bench, a["baseline_costs"][-1]) if a else "-",
            _cell(bench, a["reference_cost"]) if a else "-",
            f"{points['scores'][r.name]:.3f}" if points else "-",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        out("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
        if i == 0:
            out("  ".join("-" * w for w in widths))

    for r in records:
        notes = []
        if r.rejected:
            notes.append(f"{r.rejected} rejected: {'; '.join(r.reasons)}")
        if r.error:
            notes.append(r.error.strip().splitlines()[-1][:200])
        if notes:
            out(f"  {r.name}: " + " | ".join(notes))

    if points:
        out("")
        out(f"Points  quality {points['quality']:5.1f}/70   robustness {points['robustness']:5.1f}/20"
            f"{' (halved)' if points['robustness_halved'] else ''}   "
            f"engineering {points['engineering']:4.1f}/10   total {points['total']:5.1f}/100")
    elif anchors is None:
        out("\n(no anchors found: costs only, no scores)")


def to_json(bench, adapter, records, points, budget_scale=1.0) -> dict:
    return {
        "bench": bench.name,
        "adapter": adapter,
        "budget_scale": budget_scale,
        "records": [asdict(r) for r in records],
        "points": points,
    }
