"""Instances -> families -> 100 points."""

import json
from pathlib import Path
from statistics import fmean

from benchkit.scoring import instance_score

ANCHOR_SCHEMA = 1


class StaleAnchorsError(Exception):
    pass


def load_anchors(path, bench) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != ANCHOR_SCHEMA:
        raise StaleAnchorsError(f"{path}: anchor schema {doc.get('schema')} != {ANCHOR_SCHEMA}")
    if doc.get("generator_schema") != bench.data.SCHEMA_VERSION:
        raise StaleAnchorsError(
            f"{path}: generator schema {doc.get('generator_schema')} != {bench.data.SCHEMA_VERSION}")
    return doc


def aggregate(records, anchors: dict) -> dict:
    """Score records against an anchors document. Every record needs a fresh anchor."""
    scores = {}
    for r in records:
        anchor = anchors["instances"].get(r.name)
        if anchor is None:
            raise StaleAnchorsError(f"no anchor for {r.name}")
        # r.digest was computed by the parent from a freshly generated instance.
        if anchor["digest"] != r.digest:
            raise StaleAnchorsError(f"anchor digest for {r.name} is stale")
        scores[r.name] = instance_score(r.checkpoint_costs, anchor)

    n = len(records)
    quality = 70 * fmean(scores.values())

    families = {}
    for r in records:
        families.setdefault(r.family, []).append(scores[r.name])
    family_means = {f: fmean(v) for f, v in families.items()}
    shifted = [scores[r.name] for r in records if r.shifted]
    robustness = 20 * fmean(shifted) if shifted else 0.0
    # Specialisation penalty: only families with 2+ instances count, so one instance
    # (e.g. in a 2-instance self-check) can never trigger it.
    counted = [fmean(v) for v in families.values() if len(v) >= 2]
    halved = len(counted) >= 2 and max(counted) >= 0.75 and min(counted) < 0.50
    if halved:
        robustness /= 2

    valid_final = sum(r.final_cost is not None for r in records) / n
    clean = sum(not (r.crashed or r.overrun) for r in records) / n
    early = sum(r.checkpoint_costs[0] is not None for r in records) / n
    engineering = 5 * valid_final + 3 * clean + 2 * early

    return {
        "scores": scores,
        "family_means": family_means,
        "robustness_halved": halved,
        "quality": quality,
        "robustness": robustness,
        "engineering": engineering,
        "total": quality + robustness + engineering,
    }
