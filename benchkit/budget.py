"""Budgets, checkpoints and their weights."""

import time

CHECKPOINTS = (0.05, 0.20, 0.50, 1.00)
CHECKPOINT_WEIGHTS = (0.10, 0.20, 0.30, 0.40)
GRACE = 0.05          # candidates after budget * (1 + GRACE) are rejected
KILL_MARGIN_S = 2.0   # child is killed at budget * (1 + GRACE) + KILL_MARGIN_S


class Budget:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def remaining(self) -> float:
        return self.seconds - self.elapsed

    @property
    def grace_limit(self) -> float:
        return self.seconds * (1 + GRACE)

    @property
    def kill_at(self) -> float:
        return self.grace_limit + KILL_MARGIN_S

    def checkpoint_times(self) -> tuple:
        """Checkpoint times in seconds; the final one includes the grace window."""
        times = [f * self.seconds for f in CHECKPOINTS]
        times[-1] = self.grace_limit
        return tuple(times)
