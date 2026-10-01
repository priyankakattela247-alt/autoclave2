"""Shared machinery for the Anvil easy-track benchmarks."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Callable, NamedTuple


class SuiteEntry(NamedTuple):
    name: str
    seed: int
    profile: dict  # must contain "family" (str) and "shifted" (bool)


def freeze_profile(profile) -> MappingProxyType:
    frozen = MappingProxyType(dict(profile))
    if not isinstance(frozen.get("family"), str) or not isinstance(frozen.get("shifted"), bool):
        raise ValueError("profile needs a str 'family' and a bool 'shifted'")
    return frozen


@dataclass(frozen=True)
class BenchConfig:
    """What benchkit needs to know about one bench; built in each bench's metrics.py.

    `data` must provide SCHEMA_VERSION, make_instance(seed, profile, name),
    compute_digest(instance), PUBLIC_SUITE, SELF_CHECK_NAMES, budget_for(instance).
    `validate(instance, candidate)` returns (cost, None) or (None, reason).
    """

    name: str
    bench_dir: Path
    data: ModuleType
    validate: Callable
    starter: str = "adapters.starter:StarterSolver"
    baseline: str = "adapters.baseline:BaselineSolver"
    reference: str = "private.reference_solver:ReferenceSolver"
    probe: str = None  # textbook solver gated at MAX_PROBE_SCORE when anchoring (organizer only)
    format_cost: Callable = str
    shift_markers: Callable = lambda profile: ""

    @property
    def public_anchor_path(self) -> Path:
        return self.bench_dir / "public_reference.json"

    @property
    def private_anchor_path(self) -> Path:
        return self.bench_dir / "private" / "private_reference.json"
