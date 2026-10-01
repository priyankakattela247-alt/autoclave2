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
    Best-improvement insertion search.

    The original version stopped at the first improving move.
    Here we inspect the most expensive/tardy batches and choose
    the BEST insertion position for the batch before accepting it.

    This is slower per pass, but much less greedy and generally
    more useful on the 70-90 batch instances.
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

    # A small beam of problematic positions keeps the 5 s budget safe.
    beam = min(n, max(12, n // 3))

    while time.perf_counter() < deadline:
        contribution = [0] * n

        current_time = 0
        previous_family = None

        for pos, b in enumerate(order):
            if previous_family is None:
                start = r[b]
            else:
                start = max(
                    current_time + setup[previous_family][f[b]],
                    r[b],
                )

            current_time = start + p[b]
            contribution[pos] = w[b] * max(0, current_time - d[b])
            previous_family = f[b]

        # Tardy/high-weight batches first.  Add a mild family/setup signal
        # so batches sitting at costly family boundaries are also considered.
        positions = sorted(
            range(n),
            key=lambda i: (
                contribution[i],
                p[order[i]],
                d[order[i]],
            ),
            reverse=True,
        )[:beam]

        best_move = None
        best_delta = 0

        for i in positions:
            if time.perf_counter() >= deadline:
                break

            batch = order[i]
            reduced = order[:i] + order[i + 1:]
            batch_family = f[batch]

            # Candidate positions near the same family first, then all
            # remaining positions.  This helps reduce setup changes.
            family_positions = []
            for j in range(len(reduced) + 1):
                left = reduced[j - 1] if j > 0 else None
                right = reduced[j] if j < len(reduced) else None

                if (
                    (left is not None and f[left] == batch_family)
                    or
                    (right is not None and f[right] == batch_family)
                ):
                    family_positions.append(j)

            # Also inspect positions immediately before/after high-impact
            # jobs; these often matter more than arbitrary positions.
            important_positions = set(family_positions)

            for j in range(max(0, i - 4), min(len(reduced) + 1, i + 5)):
                important_positions.add(j)

            # Every 3rd pass / when beam is small, sample a few extra positions.
            for j in range(0, len(reduced) + 1, max(1, n // 12)):
                important_positions.add(j)

            candidate_positions = sorted(
                important_positions,
                key=lambda j: (
                    0
                    if j in important_positions and (
                        j in family_positions
                    )
                    else 1,
                    abs(j - i),
                ),
            )

            # Always include the ends.
            if 0 not in important_positions:
                candidate_positions.append(0)
            if len(reduced) not in important_positions:
                candidate_positions.append(len(reduced))

            seen = set()

            for j in candidate_positions:
                if j in seen:
                    continue
                seen.add(j)

                # Same position as before removal.
                if j == i:
                    continue

                candidate = reduced[:j] + [batch] + reduced[j:]
                candidate_cost = cost_of(instance, candidate)
                delta = current_cost - candidate_cost

                if delta > best_delta:
                    best_delta = delta
                    best_move = candidate

        if best_move is None:
            break

        order = best_move
        current_cost -= best_delta

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

def destroy_repair(instance, order, rng, deadline=None):
    """
    Large-neighborhood search with regret-2 repair.

    Compared with simple random reinsertion, regret repair asks:
        "Which removed batch will become hardest to place later?"

    That tends to preserve difficult jobs/family transitions and is
    especially useful on the tight-heavy instances.
    """
    n = len(order)

    q = rng.randint(
        max(4, n // 16),
        min(10, max(5, n // 8)),
    )

    p = instance.proc
    r = instance.release
    d = instance.due
    w = instance.weight
    f = instance.fam
    setup = instance.setup

    # --------------------------------------------------------
    # Score current positions by weighted tardiness contribution.
    # --------------------------------------------------------
    contribution = [0] * n

    current_time = 0
    previous_family = None

    for pos, b in enumerate(order):
        if previous_family is None:
            start = r[b]
        else:
            start = max(
                current_time + setup[previous_family][f[b]],
                r[b],
            )

        current_time = start + p[b]
        contribution[pos] = w[b] * max(0, current_time - d[b])
        previous_family = f[b]

    # Remove a mixture of:
    #   - one clearly problematic batch
    #   - random batches
    # This gives exploration without destroying all structure.
    bad_positions = sorted(
        range(n),
        key=lambda i: contribution[i],
        reverse=True,
    )

    target_pool = bad_positions[:max(10, n // 4)]

    removed_positions = {
        rng.choice(target_pool)
    }

    while len(removed_positions) < q:
        removed_positions.add(rng.randrange(n))

    remaining = list(order)
    removed = []

    for i in sorted(removed_positions, reverse=True):
        removed.append(remaining.pop(i))

    # --------------------------------------------------------
    # Regret-2 repair.
    # --------------------------------------------------------
    while removed:
        if deadline is not None and time.perf_counter() >= deadline:
            # Keep the solution valid even if time expires.
            for batch in removed:
                remaining.append(batch)
            return remaining

        chosen_batch = None
        chosen_position = 0
        chosen_regret = -float("inf")
        chosen_best_cost = float("inf")

        for batch in removed:
            best_cost = float("inf")
            second_cost = float("inf")
            best_pos = 0

            for j in range(len(remaining) + 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                candidate = (
                    remaining[:j]
                    + [batch]
                    + remaining[j:]
                )

                candidate_cost = cost_of(instance, candidate)

                if candidate_cost < best_cost:
                    second_cost = best_cost
                    best_cost = candidate_cost
                    best_pos = j
                elif candidate_cost < second_cost:
                    second_cost = candidate_cost

            regret = second_cost - best_cost

            # Highest regret gets inserted first.
            if (
                regret > chosen_regret
                or (
                    regret == chosen_regret
                    and best_cost < chosen_best_cost
                )
            ):
                chosen_regret = regret
                chosen_batch = batch
                chosen_position = best_pos
                chosen_best_cost = best_cost

        removed.remove(chosen_batch)
        remaining.insert(chosen_position, chosen_batch)

    return remaining



# ============================================================
# TARGETED LARGE-NEIGHBORHOOD MOVES
# ============================================================

def critical_block_descent(instance, order, deadline):
    """
    Move a critical adjacent pair as a block.

    Unlike a full O(n^3) block search, only a handful of the most
    problematic adjacent pairs are considered. This keeps the search
    useful under the 5-second budget.
    """
    order = list(order)
    n = len(order)
    current_cost = cost_of(instance, order)

    while time.perf_counter() < deadline:
        contribution = [0] * n
        t = 0
        prev = None
        p = instance.proc
        r = instance.release
        d = instance.due
        w = instance.weight
        f = instance.fam
        setup = instance.setup

        for pos, b in enumerate(order):
            start = r[b] if prev is None else max(t + setup[prev][f[b]], r[b])
            t = start + p[b]
            contribution[pos] = w[b] * max(0, t - d[b])
            prev = f[b]

        # Candidate adjacent pairs: prioritize pairs containing tardy jobs
        # and pairs crossing a family boundary (where setup matters).
        pair_score = []
        for i in range(n - 1):
            a, b = order[i], order[i + 1]
            boundary = setup[f[a]][f[b]] if f[a] != f[b] else 0
            score = contribution[i] + contribution[i + 1] + 2.0 * boundary
            pair_score.append((score, i))

        pair_score.sort(reverse=True)
        pair_positions = [i for _, i in pair_score[:min(6, n - 1)]]

        best_move = None
        best_delta = 0

        for i in pair_positions:
            if time.perf_counter() >= deadline:
                break

            block = order[i:i + 2]
            reduced = order[:i] + order[i + 2:]

            # Prefer positions adjacent to either block family, then all
            # positions. Always test ends as well.
            preferred = set()
            fam_a = f[block[0]]
            fam_b = f[block[1]]
            for j in range(len(reduced) + 1):
                left = reduced[j - 1] if j > 0 else None
                right = reduced[j] if j < len(reduced) else None
                if (
                    (left is not None and f[left] in (fam_a, fam_b))
                    or
                    (right is not None and f[right] in (fam_a, fam_b))
                ):
                    preferred.add(j)

            # Add positions near the original location and regular samples.
            for j in range(max(0, i - 5), min(len(reduced) + 1, i + 6)):
                preferred.add(j)
            step = max(1, n // 10)
            for j in range(0, len(reduced) + 1, step):
                preferred.add(j)
            preferred.add(0)
            preferred.add(len(reduced))

            for j in sorted(preferred):
                if time.perf_counter() >= deadline:
                    break
                candidate = reduced[:j] + block + reduced[j:]
                c = cost_of(instance, candidate)
                delta = current_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = candidate

        if best_move is None:
            break

        order = best_move
        current_cost -= best_delta

    return order, current_cost


def critical_swap_descent(instance, order, deadline):
    """Target pair swaps toward tardy/high-weight and setup-critical jobs."""
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
        contribution = [0] * n
        t = 0
        prev = None
        for pos, b in enumerate(order):
            start = r[b] if prev is None else max(t + setup[prev][f[b]], r[b])
            t = start + p[b]
            contribution[pos] = w[b] * max(0, t - d[b])
            prev = f[b]

        critical = sorted(range(n), key=lambda i: contribution[i], reverse=True)
        critical = critical[:min(14, n)]

        candidates = set()
        for i in critical:
            candidates.add(i)
            # Nearby positions often expose beneficial local swaps.
            for j in range(max(0, i - 4), min(n, i + 5)):
                candidates.add(j)

        best_move = None
        best_delta = 0

        for i in sorted(candidates):
            if time.perf_counter() >= deadline:
                break
            for j in sorted(candidates):
                if j <= i:
                    continue
                order[i], order[j] = order[j], order[i]
                c = cost_of(instance, order)
                delta = current_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = list(order)
                order[i], order[j] = order[j], order[i]

        if best_move is None:
            break

        order = best_move
        current_cost -= best_delta

    return order, current_cost


def family_run_descent(instance, order, deadline):
    """
    Reorder short same-family runs using several sensible rules.

    Setup is zero within a family, so improving the order inside a run
    can reduce weighted tardiness without increasing changeover count.
    """
    order = list(order)
    n = len(order)
    current_cost = cost_of(instance, order)

    while time.perf_counter() < deadline:
        improved = False
        i = 0

        while i < n and time.perf_counter() < deadline:
            j = i + 1
            fam = instance.fam[order[i]]
            while j < n and instance.fam[order[j]] == fam:
                j += 1

            if j - i >= 2:
                block = order[i:j]
                rules = [
                    lambda b: (instance.due[b], instance.release[b], b),
                    lambda b: (instance.due[b] - instance.proc[b], instance.due[b], b),
                    lambda b: (instance.due[b], -instance.weight[b], b),
                    lambda b: ((instance.due[b] - instance.release[b]) / (instance.weight[b] + 0.5), b),
                ]

                best_block = block
                best_block_cost = current_cost

                for rule in rules:
                    candidate = sorted(block, key=rule)
                    test = order[:i] + candidate + order[j:]
                    c = cost_of(instance, test)
                    if c < best_block_cost:
                        best_block = candidate
                        best_block_cost = c

                if best_block_cost < current_cost:
                    order = order[:i] + best_block + order[j:]
                    current_cost = best_block_cost
                    improved = True
                    break

            i = j

        if not improved:
            break

    return order, current_cost


def perturb_order(order, rng, strength=0.10):
    """Small random kick used between local-search basins."""
    out = list(order)
    moves = max(2, int(len(out) * strength))
    for _ in range(moves):
        i = rng.randrange(len(out))
        j = rng.randrange(len(out))
        out[i], out[j] = out[j], out[i]
    return out

# ============================================================
# SOLVER
# ============================================================

class MySolver(Solver):

    def solve(self, instance, submit_candidate):
        start_time = time.perf_counter()
        deadline = start_time + TIME_LIMIT

        try:
            seed = int(instance.digest, 16)
        except Exception:
            seed = 123456789
        rng = random.Random(seed)
        n = instance.size

        # Keep V2's excellent initial pool unchanged.
        initial_orders = [
            order_edd(instance),
            order_release(instance),
            order_mixed(instance),
            order_mdd(instance, weighted=False),
            order_mdd(instance, weighted=True),
        ]

        best_order = min(
            initial_orders,
            key=lambda x: cost_of(instance, x),
        )
        best_order = list(best_order)
        best_cost = cost_of(instance, best_order)

        # Submit immediately, preserving the benchmark's early checkpoint.
        receipt = submit_candidate({"order": best_order})
        remaining_s = receipt.get("remaining_s", TIME_LIMIT)
        deadline = min(
            deadline,
            time.perf_counter() + max(0.0, remaining_s - 0.06),
        )

        # Phase 1: the same strong single-insertion improvement that made V2
        # competitive. Spend less time on starts after we already have a very
        # good solution, especially on large instances.
        start_budget = 0.55 if n >= 70 else 0.70

        for initial in initial_orders:
            if time.perf_counter() >= deadline:
                break

            local_deadline = min(
                deadline,
                time.perf_counter() + start_budget,
            )

            candidate, candidate_cost = insertion_descent(
                instance,
                list(initial),
                local_deadline,
            )

            if candidate_cost < best_cost:
                receipt = submit_candidate({"order": candidate})
                if receipt.get("accepted", True):
                    best_order = list(candidate)
                    best_cost = candidate_cost
                    remaining_s = receipt.get("remaining_s", TIME_LIMIT)
                    deadline = min(
                        deadline,
                        time.perf_counter() + max(0.0, remaining_s - 0.06),
                    )

        # Extra large-instance refinements only. This protects the small
        # public cases where V2 already reaches the reference.
        if n >= 70 and time.perf_counter() < deadline:
            for refine_fn, seconds in (
                (family_run_descent, 0.16),
                (critical_swap_descent, 0.22),
                (critical_block_descent, 0.28),
                (insertion_descent, 0.28),
            ):
                if time.perf_counter() >= deadline:
                    break
                local_deadline = min(
                    deadline,
                    time.perf_counter() + seconds,
                )
                candidate, c = refine_fn(
                    instance,
                    list(best_order),
                    local_deadline,
                )
                if c < best_cost:
                    receipt = submit_candidate({"order": candidate})
                    if receipt.get("accepted", True):
                        best_order = list(candidate)
                        best_cost = c

        # --------------------------------------------------------
        # Main iterated search.
        # V2 always destroyed the BEST solution. Here we keep a separate
        # current solution and occasionally accept a slightly worse basin,
        # which gives the search a way to escape local minima.
        # --------------------------------------------------------
        current = list(best_order)
        current_cost = best_cost
        iteration = 0

        while time.perf_counter() < deadline:
            iteration += 1

            if n >= 70:
                destroy_low = max(5, n // 14)
                destroy_high = max(destroy_low + 1, min(11, n // 7))
            else:
                destroy_low = 3
                destroy_high = 6

            # Most iterations: destroy/repair from current rather than only
            # from best. Every few iterations jump from the global best.
            anchor = (
                best_order
                if iteration % 5 == 0
                else current
            )

            # Temporarily use V2 regret-2 destroy/repair.
            candidate = list(anchor)
            # The existing function chooses its own destroy size, so use a
            # random kick before it to diversify the destroyed set.
            if iteration % 3 == 0:
                candidate = perturb_order(candidate, rng, 0.06 if n >= 70 else 0.04)

            repair_deadline = min(
                deadline,
                time.perf_counter() + (0.24 if n >= 70 else 0.28),
            )
            candidate = destroy_repair(
                instance,
                candidate,
                rng,
                deadline=repair_deadline,
            )

            local_deadline = min(
                deadline,
                time.perf_counter() + (0.30 if n >= 70 else 0.34),
            )
            candidate, c = insertion_descent(
                instance,
                candidate,
                local_deadline,
            )

            # On large/tight cases, use one focused stronger neighborhood.
            if n >= 70 and time.perf_counter() < deadline:
                remain = deadline - time.perf_counter()
                refine_seconds = min(0.18, max(0.0, remain))
                if iteration % 3 == 0:
                    candidate, c = critical_block_descent(
                        instance,
                        candidate,
                        time.perf_counter() + refine_seconds,
                    )
                elif iteration % 3 == 1:
                    candidate, c = critical_swap_descent(
                        instance,
                        candidate,
                        time.perf_counter() + refine_seconds,
                    )
                else:
                    candidate, c = family_run_descent(
                        instance,
                        candidate,
                        time.perf_counter() + refine_seconds,
                    )

            # Global best update.
            if c < best_cost:
                receipt = submit_candidate({"order": candidate})
                if receipt.get("accepted", True):
                    best_order = list(candidate)
                    best_cost = c
                    current = list(candidate)
                    current_cost = c
                    remaining_s = receipt.get("remaining_s", TIME_LIMIT)
                    deadline = min(
                        deadline,
                        time.perf_counter() + max(0.0, remaining_s - 0.06),
                    )
                    continue

            # Diversification: accept a modestly worse candidate as current.
            if c < current_cost:
                current = list(candidate)
                current_cost = c
            else:
                delta = c - current_cost
                temperature = max(30.0, best_cost * 0.0035)
                probability = math.exp(-min(25.0, delta / temperature))
                if rng.random() < 0.08 * probability:
                    current = list(candidate)
                    current_cost = c

        return {"order": best_order}
