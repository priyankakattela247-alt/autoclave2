"""Child-process execution: the one place where isolation is enforced.

The parent owns the instance, the clock, the validator and the kill switch.
The child rebuilds the instance from its suite entry, so it never holds the
parent's object. Child -> parent messages are JSON bytes only (never pickle),
size-capped before decoding.
"""
import subprocess
import json
import multiprocessing
import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from benchkit.budget import Budget
from benchkit.loader import check_spec, load_adapter

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
SUBMISSION_CAP = 20_000
READY_TIMEOUT_S = 60.0
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3
MAX_REASONS = 5
SOLVER_UID = 65534  # nobody: the solver's uid when the harness runs as root


class HarnessIntegrityError(Exception):
    """The parent's own instance changed during a run: a harness bug, never a contestant score."""


@dataclass
class RunRecord:
    name: str
    family: str
    shifted: bool
    size: int
    digest: str
    adapter: str
    budget_s: float
    history: list = field(default_factory=list)          # [(t, cost)] improvements only
    checkpoint_costs: list = field(default_factory=list)  # best cost at each checkpoint, or None
    final_cost: int = None
    accepted: int = 0
    rejected: int = 0
    reasons: list = field(default_factory=list)          # first distinct rejection reasons
    overrun: bool = False
    crashed: bool = False
    error: str = None
    wall_s: float = 0.0


# ---------------------------------------------------------------- child side

def _encode(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


def _encode_payload(kind: str, value) -> bytes:
    try:
        body = _encode({"t": kind, "c": value})
    except (TypeError, ValueError, RecursionError) as exc:
        return _encode({"t": kind, "bad": f"candidate is not JSON-serialisable ({exc})"[:200]})
    if len(body) > MAX_MESSAGE_BYTES:
        return _encode({"t": kind, "bad": f"candidate exceeds {MAX_MESSAGE_BYTES} bytes"})
    return body


def _child_main(conn, bench_dir, adapter_spec, entry, allow_private):
    try:
        # Unix uses process groups; Windows does not provide os.setsid().
        if hasattr(os, "setsid"):
            os.setsid()  # own process group, so the parent can kill grandchildren too
    except OSError:
        pass
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass
    if hasattr(os, "getuid") and os.getuid() == 0:
        # Judging container: never run contestant code as root. As nobody the solver
        # cannot read root-only files (private/), write the report, or signal the parent.
        try:
            os.setgroups([])
            os.setgid(SOLVER_UID)
            os.setuid(SOLVER_UID)
        except OSError as exc:
            conn.send_bytes(_encode({"t": "error", "tb": f"could not drop privileges: {exc}"}))
            return

    try:
        sys.path.insert(0, bench_dir)
        import importlib
        data = importlib.import_module("data")
        name, seed, profile = entry
        instance = data.make_instance(seed, profile, name)
        solver = load_adapter(adapter_spec, bench_dir, allow_private=allow_private)()
    except BaseException:
        conn.send_bytes(_encode({"t": "error", "tb": traceback.format_exc(limit=5)}))
        return
    conn.send_bytes(_encode({"t": "ready"}))
    conn.recv_bytes()  # "go"

    def submit_candidate(candidate):
        conn.send_bytes(_encode_payload("cand", candidate))
        return json.loads(conn.recv_bytes())

    try:
        result = solver.solve(instance, submit_candidate)
    except BaseException:
        conn.send_bytes(_encode({"t": "error", "tb": traceback.format_exc(limit=5)}))
        return
    conn.send_bytes(_encode({"t": "final"}) if result is None else _encode_payload("final", result))


# --------------------------------------------------------------- parent side

class _ProtocolError(Exception):
    pass


class _ChildFailed(Exception):
    pass


def _reject_constant(name):
    raise ValueError(f"non-finite number {name}")


def _decode(raw: bytes) -> dict:
    try:
        msg = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise _ProtocolError(f"undecodable message ({type(exc).__name__})") from None
    if not isinstance(msg, dict) or msg.get("t") not in ("ready", "cand", "final", "error"):
        raise _ProtocolError("malformed message")
    return msg


def _kill(proc):
    if proc.is_alive() is not None:
        return

    if hasattr(os, "killpg"):
        # Linux / macOS
        os.killpg(proc.pid, signal.SIGKILL)
    else:
        # Windows
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def costs_at(history, times) -> list:
    return [min((c for t, c in history if t <= ct), default=None) for ct in times]


def run_instance(bench, entry, adapter_spec, budget_s=None, allow_private=False) -> RunRecord:
    """Run one adapter on one suite entry in a child process and return its record."""
    check_spec(adapter_spec, allow_private)
    instance = bench.data.make_instance(entry.seed, entry.profile, entry.name)
    if budget_s is None:
        budget_s = bench.data.budget_for(instance)
    budget = Budget(budget_s)

    # Captured before the child can run anything.
    profile_before = dict(instance.profile)
    digest_before = instance.digest

    record = RunRecord(
        name=instance.name, family=profile_before["family"], shifted=profile_before["shifted"],
        size=instance.size, digest=digest_before, adapter=adapter_spec, budget_s=budget.seconds)

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(
        target=_child_main,
        args=(child_conn, str(Path(bench.bench_dir).resolve()), adapter_spec,
              (entry.name, entry.seed, dict(entry.profile)), allow_private),
        daemon=True)
    wall = Budget(0)
    wall.start()
    proc.start()
    child_conn.close()

    best = None
    submissions = 0

    def reject(reason):
        record.rejected += 1
        if reason not in record.reasons and len(record.reasons) < MAX_REASONS:
            record.reasons.append(reason)

    def consider(msg) -> dict:
        nonlocal best, submissions
        submissions += 1
        cost = None
        if submissions > SUBMISSION_CAP:
            reason = f"submission cap of {SUBMISSION_CAP} reached"
        elif "bad" in msg:
            reason = str(msg["bad"])[:200]
        elif "c" not in msg:
            reason = "empty candidate"
        else:
            try:
                cost, reason = bench.validate(instance, msg["c"])
            except Exception as exc:  # validators must not take the harness down
                cost, reason = None, f"validator error: {type(exc).__name__}: {exc}"[:200]
        t = budget.elapsed  # taken after validation finishes
        if cost is not None and t > budget.grace_limit:
            cost, reason = None, "submitted after the grace window"
        if cost is None:
            reject(reason or "rejected")
        else:
            record.accepted += 1
            if best is None or cost < best:
                best = cost
                record.history.append((round(t, 6), cost))
        return {"accepted": cost is not None, "reason": reason, "cost": cost, "best": best,
                "elapsed_s": round(t, 4), "remaining_s": round(budget.seconds - t, 4)}

    try:
        # Wait for the child to import the adapter and build the solver.
        if not parent_conn.poll(READY_TIMEOUT_S):
            raise _ProtocolError("child did not become ready")
        msg = _decode(parent_conn.recv_bytes(MAX_MESSAGE_BYTES))
        if msg["t"] == "error":
            raise _ChildFailed(msg.get("tb", ""))
        if msg["t"] != "ready":
            raise _ProtocolError("expected ready")

        budget.start()
        parent_conn.send_bytes(_encode({"t": "go"}))
        while True:
            remaining = budget.kill_at - budget.elapsed
            if remaining <= 0 or not parent_conn.poll(remaining):
                if budget.elapsed >= budget.kill_at:
                    record.overrun = True
                    record.error = "killed after budget + grace + kill margin"
                    break
                continue
            msg = _decode(parent_conn.recv_bytes(MAX_MESSAGE_BYTES))
            if msg["t"] == "cand":
                parent_conn.send_bytes(_encode(consider(msg)))
            elif msg["t"] == "final":
                if "c" in msg or "bad" in msg:
                    consider(msg)
                break
            elif msg["t"] == "error":
                raise _ChildFailed(msg.get("tb", ""))
            else:
                raise _ProtocolError("unexpected message")
    except _ChildFailed as exc:
        record.crashed = True
        record.error = str(exc)[-2000:]
    except EOFError:
        record.crashed = True
        record.error = "child exited without a result"
    except (OSError, _ProtocolError) as exc:
        # recv_bytes raises OSError when a message exceeds MAX_MESSAGE_BYTES.
        record.crashed = True
        record.error = f"protocol violation: {exc}"
    finally:
        _kill(proc)
        parent_conn.close()
        record.wall_s = round(wall.elapsed, 3)

    if dict(instance.profile) != profile_before or bench.data.compute_digest(instance) != digest_before:
        raise HarnessIntegrityError(f"instance {instance.name} changed during the run")
    record.final_cost = best
    record.checkpoint_costs = costs_at(record.history, budget.checkpoint_times())
    return record
