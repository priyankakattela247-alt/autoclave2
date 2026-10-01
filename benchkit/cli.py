"""Shared entry points behind each bench's self_check.py, run.py and private/make_reference.py."""

import argparse
import importlib
import json
import os
import sys

from benchkit import anchors as anchor_gen
from benchkit.aggregate import StaleAnchorsError, aggregate, load_anchors
from benchkit.loader import LoaderError
from benchkit.report import print_table, to_json
from benchkit.runner import run_instance


def _private_suite(bench):
    if not (bench.bench_dir / "private" / "private_suite.py").exists():
        raise SystemExit("the private suite is not available in this tree")
    return importlib.import_module("private.private_suite").PRIVATE_SUITE


def _run(bench, entries, adapter, budget_scale, allow_private=False):
    records = []
    for entry in entries:
        instance = bench.data.make_instance(entry.seed, entry.profile, entry.name)
        budget = bench.data.budget_for(instance) * budget_scale
        print(f"running {entry.name} ({budget:.1f} s) ...", file=sys.stderr, flush=True)
        records.append(run_instance(bench, entry, adapter, budget, allow_private=allow_private))
    return records


def _score(bench, records, anchor_path):
    if not anchor_path.exists():
        return None, None
    anchors = load_anchors(anchor_path, bench)
    return anchors, aggregate(records, anchors)


def self_check_main(bench, argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{bench.name}: quick check on 2 public instances")
    ap.add_argument("--adapter", default=bench.starter, help="module:Class (default: the starter)")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    entries = [e for e in bench.data.PUBLIC_SUITE if e.name in bench.data.SELF_CHECK_NAMES]
    try:
        records = _run(bench, entries, args.adapter, args.budget_scale)
        anchors, points = _score(bench, records, bench.public_anchor_path)
    except (LoaderError, StaleAnchorsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_table(bench, records, anchors, points)
    return 0


def run_main(bench, argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{bench.name}: full suite")
    ap.add_argument("--adapter", default=bench.starter, help="module:Class (default: the starter)")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--suite", choices=("public", "private"), default="public")
    ap.add_argument("--allow-private", action="store_true",
                    help="organizer only: allow adapters under private/ (probes)")
    args = ap.parse_args(argv)
    if args.allow_private and not (bench.bench_dir / "private").is_dir():
        print("error: --allow-private needs the organizer tree (no private/ here)", file=sys.stderr)
        return 2
    if args.suite == "private":
        entries, anchor_path = _private_suite(bench), bench.private_anchor_path
    else:
        entries, anchor_path = bench.data.PUBLIC_SUITE, bench.public_anchor_path
    try:
        records = _run(bench, entries, args.adapter, args.budget_scale, args.allow_private)
        anchors, points = _score(bench, records, anchor_path)
    except (LoaderError, StaleAnchorsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_table(bench, records, anchors, points)
    with open(args.out, "w") as f:
        json.dump(to_json(bench, args.adapter, records, points, args.budget_scale), f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


def make_reference_main(bench, argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{bench.name}: generate anchors (run on an idle machine)")
    ap.add_argument("--quick", action="store_true",
                    help=f"1 repeat, reference at {anchor_gen.QUICK_REFERENCE_SCALE}x, instances in "
                         "parallel; print only, write nothing")
    ap.add_argument("--suite", choices=("public", "private", "both"), default="both")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None,
                    help="instances run at once (default 1; real anchors must use 1)")
    args = ap.parse_args(argv)
    repeats = 1 if args.quick else args.repeats
    scale = anchor_gen.QUICK_REFERENCE_SCALE if args.quick else anchor_gen.REFERENCE_SCALE
    workers = args.workers or (max(1, (os.cpu_count() or 2) // 2) if args.quick else 1)

    targets = []
    if args.suite in ("public", "both"):
        targets.append(("public", bench.data.PUBLIC_SUITE, bench.public_anchor_path))
    if args.suite in ("private", "both"):
        targets.append(("private", _private_suite(bench), bench.private_anchor_path))

    failed = False
    for label, entries, path in targets:
        print(f"[{label}] {len(entries)} instances, {repeats} repeat(s), reference at {scale}x, "
              f"{workers} worker(s)")
        doc, problems = anchor_gen.generate(bench, entries, repeats, print, scale, workers)
        print(f"\n{'instance':<16} {'starter':>12} {'baseline':>12} {'reference':>12}")
        for name, a in doc["instances"].items():
            print(f"{name:<16} {str(a['starter_costs'][-1]):>12} {str(a['baseline_costs'][-1]):>12} "
                  f"{str(a['reference_cost']):>12}")
        for name, issues in problems.items():
            for issue in issues:
                print(f"PROBLEM {name}: {issue}")
        if problems:
            failed = True
            if not args.quick:
                print(f"not writing {path}")
                continue
        if not args.quick:
            anchor_gen.write(doc, path)
            print(f"wrote {path}")
    return 1 if failed else 0


def find_attempts_main(bench, accept, argv=None, max_attempts=200) -> int:
    """Organizer tool behind each bench's private/find_attempts.py.

    `accept(seed, profile)` returns (ok, note) for the draw selected by
    profile["attempt"]. With --check, verifies the stored attempts; otherwise
    searches attempts from 0 and reports the first accepted one per entry.
    """
    ap = argparse.ArgumentParser(description=f"{bench.name}: redraw search for suite attempts")
    ap.add_argument("--check", action="store_true", help="only verify the stored attempts")
    args = ap.parse_args(argv)
    entries = list(bench.data.PUBLIC_SUITE) + list(_private_suite(bench))
    bad = 0
    for e in entries:
        stored = e.profile.get("attempt")
        if args.check:
            ok, note = accept(e.seed, e.profile)
            print(f"{e.name:<14} attempt {stored:<4} {'ok ' if ok else 'FAIL'}  {note}")
            bad += not ok
            continue
        for a in range(max_attempts):
            ok, note = accept(e.seed, {**e.profile, "attempt": a})
            if ok:
                break
        else:
            print(f"{e.name:<14} no accepted attempt below {max_attempts}")
            bad += 1
            continue
        flag = "" if a == stored else f"   <- CHANGE (stored {stored})"
        print(f"{e.name:<14} attempt {a:<4} {note}{flag}")
        bad += a != stored
    return 1 if bad else 0
