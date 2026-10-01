"""
Strong Autoclave Queue solver.

Strategy:
1. Generate several good initial permutations.
2. Submit the best immediately.
3. Insertion local search:
      remove one batch -> try inserting it at every position.
4. Destroy + repair:
      remove several batches -> greedily reinsert them.
5. Randomized restarts / perturbations to escape local optima.
6. Occasional pair-swap descent.
7. Stay within the benchmark's ~5 second budget.

The evaluator expects:
    {"order": [all batch ids exactly once]}
"""

from adapter import Solver, cost_of

import math
import random
import time


# The benchmark gives 5 seconds.
# Leave a small safety margin for submission/evaluator overhead.
TIME_LIMIT = 4.85


# ============================================================
# BASIC CONSTRUCTION HEURISTICS
# ============================================================

def order_edd(instance):
    """Earliest Due Date."""
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (
            instance.due[b],
            instance.release[b],
            b
        )
    )


def order_release(instance):
    """
    Release-time driven ordering.

    Particularly useful for the shifted/tight instances where
    release times contain useful information about the hidden
    construction sequence.
    """
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (
            instance.release[b],
            instance.due[b],
            -instance.weight[b],
            b
        )
    )


def order_mixed(instance):
    """Mix release time and due date."""
    n = instance.size

    return sorted(
        range(n),
        key=lambda b: (
            instance.release[b] + instance.due[b],
            instance.due[b],
            -instance.weight[b],
            b
        )
    )


def order_mdd(instance, weighted=False):
    """
    Modified Due Date style dispatching rule.

    At every step, choose the batch with the smallest projected
    priority after considering:
        - current machine time
        - release time
        - setup time
        - processing time
        - due date
    """
    n = instance.size

    p = instance.proc
    r = instance.release
    d = instance.due
    w = instance.weight
    f = instance.fam
    setup = instance.setup

    remaining = set(range(n))
    order = []

    current_time = 0
    previous_family = None

    while remaining:

        best_batch = None
        best_score = float("inf")

        for b in remaining:

            changeover = (
                0
                if previous_family is None
                else setup[previous_family][f[b]]
            )

            start = max(
                current_time + changeover,
                r[b]
            )

            finish = start + p[b]

            if weighted:
                # Weighted modified due date.
                score = max(finish, d[b]) / (w[b] + 0.5)
            else:
                score = max(finish, d[b])

            # Slightly discourage expensive family changes.
            score += 0.20 * changeover

            if score < best_score:
                best_score = score
                best_batch = b

        b = best_batch

        changeover = (
            0
            if previous_family is None
            else setup[previous_family][f[b]]
        )

        current_time = (
            max(current_time + changeover, r[b])
            + p[b]
        )

        previous_family = f[b]
        order.append(b)
        remaining.remove(b)

    return order


# ============================================================
# INSERTION LOCAL SEARCH
# ============================================================

def insertion_descent(instance, order, deadline):
    """
    Repeatedly perform:

        batch at position i
              |
              v
        remove it
              |
              v
        try every insertion position j

    Accept the first improving move.

    This is the most important improvement over the supplied
    baseline, which mainly relies on adjacent swaps.
    """

    order = list(order)
    n = len(order)

    current_cost = cost_of(instance, order)

    p = instance.proc
    r = instance.release
    d = instance.due
    w = instance.weight
    f = instance.fam
    setup = instance.setup

    while time.perf_counter() < deadline:

        # ----------------------------------------------------
        # Find which jobs are currently causing the most cost.
        # Try those first.
        # ----------------------------------------------------

        contribution = [0] * n

        current_time = 0
        previous_family = None

        for pos, b in enumerate(order):

            if previous_family is None:
                start = r[b]
            else:
                start = max(
                    current_time + setup[previous_family][f[b]],
                    r[b]
                )

            current_time = start + p[b]

            lateness = max(0, current_time - d[b])

            contribution[pos] = w[b] * lateness

            previous_family = f[b]

        positions = sorted(
            range(n),
            key=lambda i: contribution[i],
            reverse=True
        )

        improved = False

        # ----------------------------------------------------
        # Try removing one batch and reinserting it elsewhere.
        # ----------------------------------------------------

        for i in positions:

            if time.perf_counter() >= deadline:
                break

            batch = order[i]

            reduced = order[:i] + order[i + 1:]

            batch_family = f[batch]

            # Try positions next to the same family first.
            family_positions = []

            for j in range(len(reduced) + 1):

                left = reduced[j - 1] if j > 0 else None
                right = (
                    reduced[j]
                    if j < len(reduced)
                    else None
                )

                if (
                    left is not None
                    and f[left] == batch_family
                ) or (
                    right is not None
                    and f[right] == batch_family
                ):
                    family_positions.append(j)

            seen = set(family_positions)

            positions_to_test = (
                family_positions
                + [
                    j
                    for j in range(len(reduced) + 1)
                    if j not in seen
                ]
            )

            for j in positions_to_test:

                # This reproduces the original sequence.
                if j == i:
                    continue

                candidate = (
                    reduced[:j]
                    + [batch]
                    + reduced[j:]
                )

                candidate_cost = cost_of(
                    instance,
                    candidate
                )

                if candidate_cost < current_cost:

                    order = candidate
                    current_cost = candidate_cost

                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return order, current_cost


# ============================================================
# SWAP LOCAL SEARCH
# ============================================================

def swap_descent(instance, order, deadline):
    """
    Additional neighborhood:

        swap batch i with batch j

    Insertion search already covers many swaps, but explicit
    pair swaps can escape some insertion-local optima.
    """

    order = list(order)
    n = len(order)

    current_cost = cost_of(instance, order)

    while time.perf_counter() < deadline:

        improved = False

        for i in range(n - 1):

            for j in range(i + 1, n):

                if time.perf_counter() >= deadline:
                    return order, current_cost

                order[i], order[j] = order[j], order[i]

                new_cost = cost_of(
                    instance,
                    order
                )

                if new_cost < current_cost:

                    current_cost = new_cost
                    improved = True
                    break

                # Undo unsuccessful swap.
                order[i], order[j] = order[j], order[i]

            if improved:
                break

        if not improved:
            break

    return order, current_cost


# ============================================================
# DESTROY + REPAIR
# ============================================================

def destroy_repair(instance, order, rng):
    """
    Large-neighborhood move.

    1. Remove several batches.
    2. Reinsert them one by one in their best positions.

    This is useful because simple insertion descent can get stuck
    when several batches need to move together.
    """

    n = len(order)

    # Moderate destroy size.
    q = rng.randint(
        3,
        min(7, max(3, n // 10))
    )

    # --------------------------------------------------------
    # Calculate current tardiness contributions.
    # --------------------------------------------------------

    p = instance.proc
    r = instance.release
    d = instance.due
    w = instance.weight
    f = instance.fam
    setup = instance.setup

    contribution = [0] * n

    current_time = 0
    previous_family = None

    for pos, b in enumerate(order):

        if previous_family is None:
            start = r[b]
        else:
            start = max(
                current_time + setup[previous_family][f[b]],
                r[b]
            )

        current_time = start + p[b]

        contribution[pos] = (
            w[b] * max(0, current_time - d[b])
        )

        previous_family = f[b]

    # --------------------------------------------------------
    # Ensure at least one highly problematic batch is removed.
    # --------------------------------------------------------

    bad_positions = sorted(
        range(n),
        key=lambda i: contribution[i],
        reverse=True
    )

    target_pool = bad_positions[:max(8, n // 5)]

    removed_positions = {
        rng.choice(target_pool)
    }

    while len(removed_positions) < q:
        removed_positions.add(
            rng.randrange(n)
        )

    # --------------------------------------------------------
    # Remove batches.
    # --------------------------------------------------------

    removed = [
        order[i]
        for i in sorted(
            removed_positions,
            reverse=True
        )
    ]

    remaining = list(order)

    for i in sorted(
        removed_positions,
        reverse=True
    ):
        remaining.pop(i)

    # Randomize destruction order.
    rng.shuffle(removed)

    # --------------------------------------------------------
    # Greedy best insertion repair.
    # --------------------------------------------------------

    for batch in removed:

        best_position = 0
        best_cost = float("inf")

        for j in range(len(remaining) + 1):

            candidate = (
                remaining[:j]
                + [batch]
                + remaining[j:]
            )

            candidate_cost = cost_of(
                instance,
                candidate
            )

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_position = j

        remaining.insert(
            best_position,
            batch
        )

    return remaining


# ============================================================
# SOLVER
# ============================================================

class MySolver(Solver):

    def solve(self, instance, submit_candidate):

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Initial deadline.
        # ----------------------------------------------------

        deadline = start_time + TIME_LIMIT

        # Deterministic random generator per instance.
        try:
            seed = int(instance.digest, 16)
        except Exception:
            seed = 123456789

        rng = random.Random(seed)

        n = instance.size

        # ----------------------------------------------------
        # Generate several structurally different starting
        # solutions.
        # ----------------------------------------------------

        initial_orders = [
            order_edd(instance),
            order_release(instance),
            order_mixed(instance),
            order_mdd(instance, weighted=False),
            order_mdd(instance, weighted=True),
        ]

        # ----------------------------------------------------
        # Choose cheapest initial order.
        #
        # We do NOT spend seconds improving before submitting.
        # The first benchmark checkpoint matters.
        # ----------------------------------------------------

        best_order = min(
            initial_orders,
            key=lambda x: cost_of(instance, x)
        )

        best_cost = cost_of(
            instance,
            best_order
        )

        # ----------------------------------------------------
        # Submit first solution immediately.
        # ----------------------------------------------------

        receipt = submit_candidate({
            "order": best_order
        })

        if not receipt.get("accepted", False):
            # Still keep solving; the order itself is valid.
            pass

        # Use evaluator's remaining budget if available.
        remaining_s = receipt.get(
            "remaining_s",
            TIME_LIMIT
        )

        deadline = min(
            start_time + TIME_LIMIT,
            time.perf_counter() + max(
                0.0,
                remaining_s - 0.06
            )
        )

        # ----------------------------------------------------
        # Improve all useful starting regions.
        # ----------------------------------------------------

        for initial in initial_orders:

            if time.perf_counter() >= deadline:
                break

            # Give each start some search time.
            local_deadline = min(
                deadline,
                time.perf_counter() + 0.55
            )

            candidate = list(initial)

            candidate, candidate_cost = insertion_descent(
                instance,
                candidate,
                local_deadline
            )

            if candidate_cost < best_cost:

                receipt = submit_candidate({
                    "order": candidate
                })

                if receipt.get("accepted", True):
                    best_order = list(candidate)
                    best_cost = candidate_cost

                    remaining_s = receipt.get(
                        "remaining_s",
                        TIME_LIMIT
                    )

                    deadline = min(
                        deadline,
                        time.perf_counter()
                        + max(0.0, remaining_s - 0.06)
                    )

        # ----------------------------------------------------
        # Main iterated local search.
        # ----------------------------------------------------

        iteration = 0

        while time.perf_counter() < deadline:

            iteration += 1

            # ------------------------------------------------
            # Every few iterations, use pair-swap refinement.
            # ------------------------------------------------

            if iteration % 6 == 0:

                swap_deadline = min(
                    deadline,
                    time.perf_counter() + 0.20
                )

                candidate, candidate_cost = swap_descent(
                    instance,
                    best_order,
                    swap_deadline
                )

            else:

                # ------------------------------------------------
                # Otherwise perform a destroy/repair kick.
                # ------------------------------------------------

                candidate = destroy_repair(
                    instance,
                    best_order,
                    rng
                )

                # Follow it with insertion descent.
                candidate_deadline = min(
                    deadline,
                    time.perf_counter() + 0.35
                )

                candidate, candidate_cost = (
                    insertion_descent(
                        instance,
                        candidate,
                        candidate_deadline
                    )
                )

            # ------------------------------------------------
            # New best.
            # ------------------------------------------------

            if candidate_cost < best_cost:

                receipt = submit_candidate({
                    "order": candidate
                })

                if receipt.get("accepted", True):

                    best_order = list(candidate)
                    best_cost = candidate_cost

                    remaining_s = receipt.get(
                        "remaining_s",
                        TIME_LIMIT
                    )

                    # Refresh deadline using evaluator time.
                    deadline = min(
                        deadline,
                        time.perf_counter()
                        + max(0.0, remaining_s - 0.06)
                    )

            # ------------------------------------------------
            # Occasionally restart from a different heuristic.
            # ------------------------------------------------

            elif iteration % 10 == 0:

                restart = list(
                    initial_orders[
                        iteration % len(initial_orders)
                    ]
                )

                # Small random perturbation.
                for _ in range(
                    max(2, n // 25)
                ):

                    i = rng.randrange(n)
                    j = rng.randrange(n)

                    restart[i], restart[j] = (
                        restart[j],
                        restart[i]
                    )

                restart_deadline = min(
                    deadline,
                    time.perf_counter() + 0.25
                )

                restart, restart_cost = (
                    insertion_descent(
                        instance,
                        restart,
                        restart_deadline
                    )
                )

                if restart_cost < best_cost:

                    receipt = submit_candidate({
                        "order": restart
                    })

                    if receipt.get("accepted", True):
                        best_order = list(restart)
                        best_cost = restart_cost

        # ----------------------------------------------------
        # Final valid best solution.
        # ----------------------------------------------------

        return {
            "order": best_order
        }