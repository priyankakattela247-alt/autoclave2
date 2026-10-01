"""
Autoclave solver v3.

This version keeps the good parts of v2 and adds stronger neighborhoods:
- NEH-style insertion construction from multiple priority orders
- full/targeted best-improvement single-job insertion
- two-job block relocation
- regret-2 destroy/repair
- swap descent
- iterated perturbation with diversification

The solver never relies on the public benchmark instances.
"""

from adapter import Solver, cost_of

import math
import random
import time

TIME_LIMIT = 4.90
SAFETY = 0.07


def _seed(instance):
    try:
        return int(instance.digest, 16)
    except Exception:
        return hash(str(instance.digest)) & 0xFFFFFFFF


def order_edd(instance):
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (instance.due[b], instance.release[b], -instance.weight[b], b),
    )


def order_release(instance):
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (instance.release[b], instance.due[b], -instance.weight[b], b),
    )


def order_mixed(instance):
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (
            instance.release[b] + instance.due[b],
            instance.due[b],
            -instance.weight[b],
            b,
        ),
    )


def order_slack(instance):
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (
            instance.due[b] - instance.release[b] - instance.proc[b],
            instance.due[b],
            -instance.weight[b],
            b,
        ),
    )


def order_weighted(instance):
    n = instance.size
    return sorted(
        range(n),
        key=lambda b: (
            (instance.due[b] - instance.release[b]) / (instance.weight[b] + 0.5),
            instance.due[b],
            instance.proc[b],
            b,
        ),
    )


def order_family_due(instance):
    """Keep related families together when their due-date pressure is similar."""
    fams = {}
    for b in range(instance.size):
        fams.setdefault(instance.fam[b], []).append(b)

    family_key = {}
    for fam, jobs in fams.items():
        family_key[fam] = min(
            instance.due[b] - instance.weight[b] * 0.5
            for b in jobs
        )

    return sorted(
        range(instance.size),
        key=lambda b: (
            family_key[instance.fam[b]],
            instance.due[b],
            -instance.weight[b],
            b,
        ),
    )


def order_mdd(instance, weighted=False):
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
        best = None
        best_score = float("inf")

        for b in remaining:
            changeover = 0 if previous_family is None else setup[previous_family][f[b]]
            start = max(current_time + changeover, r[b])
            finish = start + p[b]

            if weighted:
                # Smaller is better; high-weight overdue work gets priority.
                score = max(finish, d[b]) / (w[b] + 0.5)
            else:
                score = max(finish, d[b])

            # Setup is deliberately a secondary effect, not a dominant rule.
            score += 0.18 * changeover

            if finish > d[b]:
                score -= min(25.0, 4.0 * w[b])

            if score < best_score:
                best_score = score
                best = b

        changeover = 0 if previous_family is None else setup[previous_family][f[best]]
        current_time = max(current_time + changeover, r[best]) + p[best]
        previous_family = f[best]
        order.append(best)
        remaining.remove(best)

    return order


def neh_construct(instance, seed_order):
    """NEH-style construction: insert each job in its best position."""
    order = []
    for b in seed_order:
        best_cost = float("inf")
        best_pos = 0
        for j in range(len(order) + 1):
            candidate = order[:j] + [b] + order[j:]
            c = cost_of(instance, candidate)
            if c < best_cost:
                best_cost = c
                best_pos = j
        order.insert(best_pos, b)
    return order


def randomized_order(base, rng, strength=0.10):
    """Small deterministic perturbation around a good priority order."""
    decorated = []
    for b in base:
        noise = rng.random() * strength
        decorated.append((noise, b))
    # Keep most of the original order while allowing diversified restarts.
    out = list(base)
    moves = max(2, int(len(out) * strength))
    for _ in range(moves):
        i = rng.randrange(len(out))
        j = rng.randrange(len(out))
        out[i], out[j] = out[j], out[i]
    return out


def _contributions(instance, order):
    p, r, d, w, f, setup = (
        instance.proc,
        instance.release,
        instance.due,
        instance.weight,
        instance.fam,
        instance.setup,
    )

    result = [0] * len(order)
    t = 0
    prev = None

    for pos, b in enumerate(order):
        start = r[b] if prev is None else max(t + setup[prev][f[b]], r[b])
        t = start + p[b]
        result[pos] = w[b] * max(0, t - d[b])
        prev = f[b]

    return result


def _candidate_positions(instance, reduced, batch, original_pos):
    """Prioritize family-adjacent and locally important insertion points."""
    f = instance.fam
    n = len(reduced)
    fam = f[batch]

    positions = set()

    for j in range(n + 1):
        left = reduced[j - 1] if j > 0 else None
        right = reduced[j] if j < n else None
        if (
            (left is not None and f[left] == fam)
            or (right is not None and f[right] == fam)
        ):
            positions.add(j)

    for j in range(max(0, original_pos - 5), min(n + 1, original_pos + 6)):
        positions.add(j)

    # Sparse global coverage so a good distant move isn't missed.
    stride = max(1, n // 10)
    for j in range(0, n + 1, stride):
        positions.add(j)
    positions.add(0)
    positions.add(n)

    return sorted(positions)


def insertion_descent(instance, order, deadline, beam=None):
    """Best-improvement single-batch insertion."""
    order = list(order)
    n = len(order)
    current_cost = cost_of(instance, order)

    if beam is None:
        beam = min(n, 16 if n >= 70 else max(10, n // 3))

    while time.perf_counter() < deadline:
        contributions = _contributions(instance, order)

        # Include some non-tardy jobs for setup/makespan moves.
        ranked = sorted(
            range(n),
            key=lambda i: (
                contributions[i],
                instance.weight[order[i]],
                instance.proc[order[i]],
            ),
            reverse=True,
        )
        positions = ranked[:beam]

        best_order = None
        best_cost = current_cost

        for i in positions:
            if time.perf_counter() >= deadline:
                break

            b = order[i]
            reduced = order[:i] + order[i + 1:]

            for j in _candidate_positions(instance, reduced, b, i):
                if j == i:
                    continue

                candidate = reduced[:j] + [b] + reduced[j:]
                c = cost_of(instance, candidate)

                if c < best_cost:
                    best_cost = c
                    best_order = candidate

        if best_order is None:
            break

        order = best_order
        current_cost = best_cost

    return order, current_cost


def block2_descent(instance, order, deadline):
    """Relocate two consecutive jobs together."""
    order = list(order)
    n = len(order)
    current_cost = cost_of(instance, order)

    while time.perf_counter() < deadline:
        contrib = _contributions(instance, order)
        ranked = sorted(range(n - 1), key=lambda i: contrib[i] + contrib[i + 1], reverse=True)
        ranked = ranked[: min(12, len(ranked))]

        best_order = None
        best_cost = current_cost

        for i in ranked:
            if time.perf_counter() >= deadline:
                break

            block = order[i:i + 2]
            reduced = order[:i] + order[i + 2:]

            # Global but bounded position scan.
            for j in range(len(reduced) + 1):
                candidate = reduced[:j] + block + reduced[j:]
                c = cost_of(instance, candidate)
                if c < best_cost:
                    best_cost = c
                    best_order = candidate

        if best_order is None:
            break

        order = best_order
        current_cost = best_cost

    return order, current_cost


def swap_descent(instance, order, deadline):
    order = list(order)
    n = len(order)
    current_cost = cost_of(instance, order)

    while time.perf_counter() < deadline:
        contrib = _contributions(instance, order)
        hot = sorted(range(n), key=lambda i: contrib[i], reverse=True)
        hot = hot[: min(24, n)]

        best_order = None
        best_cost = current_cost

        # Hot-hot and hot-global swaps.
        pairs = []
        for i_idx, i in enumerate(hot):
            for j in range(i_idx + 1, len(hot)):
                pairs.append((i, hot[j]))

        # A few global positions for diversification.
        for i in hot[:10]:
            for j in range(n):
                if i != j:
                    pairs.append((i, j))

        seen = set()
        for i, j in pairs:
            if i == j:
                continue
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))

            if time.perf_counter() >= deadline:
                break

            candidate = list(order)
            candidate[a], candidate[b] = candidate[b], candidate[a]
            c = cost_of(instance, candidate)

            if c < best_cost:
                best_cost = c
                best_order = candidate

        if best_order is None:
            break

        order = best_order
        current_cost = best_cost

    return order, current_cost


def destroy_repair(instance, order, rng, deadline=None):
    """Large-neighborhood search with regret-2 repair."""
    n = len(order)
    q = rng.randint(
        max(5, n // 12),
        min(12, max(6, n // 7)),
    )

    contribution = _contributions(instance, order)
    bad = sorted(range(n), key=lambda i: contribution[i], reverse=True)
    pool = bad[: max(12, n // 4)]

    removed_positions = {rng.choice(pool)}
    while len(removed_positions) < q:
        removed_positions.add(rng.randrange(n))

    remaining = list(order)
    removed = []
    for i in sorted(removed_positions, reverse=True):
        removed.append(remaining.pop(i))

    # Regret-2: insert the jobs whose second-best position is much worse
    # than their best position first.
    while removed:
        if deadline is not None and time.perf_counter() >= deadline:
            rng.shuffle(removed)
            remaining.extend(removed)
            return remaining

        chosen = None
        chosen_pos = 0
        best_regret = -float("inf")
        chosen_cost = float("inf")

        for b in removed:
            best = float("inf")
            second = float("inf")
            best_pos = 0

            for j in range(len(remaining) + 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                candidate = remaining[:j] + [b] + remaining[j:]
                c = cost_of(instance, candidate)
                if c < best:
                    second = best
                    best = c
                    best_pos = j
                elif c < second:
                    second = c

            regret = (second - best) if second < float("inf") else 0.0
            if regret > best_regret or (regret == best_regret and best < chosen_cost):
                chosen = b
                chosen_pos = best_pos
                best_regret = regret
                chosen_cost = best

        removed.remove(chosen)
        remaining.insert(chosen_pos, chosen)

    return remaining


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        start = time.perf_counter()
        deadline = start + TIME_LIMIT
        rng = random.Random(_seed(instance))
        n = instance.size

        # --------------------------------------------------------
        # Strong initial pool.
        # --------------------------------------------------------
        priority_orders = [
            order_edd(instance),
            order_release(instance),
            order_mixed(instance),
            order_slack(instance),
            order_weighted(instance),
            order_family_due(instance),
            order_mdd(instance, False),
            order_mdd(instance, True),
        ]

        initial_orders = list(priority_orders)

        # NEH is especially valuable on the larger instances.
        # Limit it to the best few priority lists to preserve time.
        neh_count = 4 if n >= 70 else 3
        neh_budget = min(deadline, time.perf_counter() + (0.80 if n >= 70 else 0.55))
        for base in priority_orders[:neh_count]:
            if time.perf_counter() >= neh_budget:
                break
            initial_orders.append(neh_construct(instance, base))

        # Best immediate candidate.
        best_order = min(initial_orders, key=lambda x: cost_of(instance, x))
        best_order = list(best_order)
        best_cost = cost_of(instance, best_order)

        receipt = submit_candidate({"order": best_order})
        remaining_s = receipt.get("remaining_s", TIME_LIMIT)
        deadline = min(
            deadline,
            time.perf_counter() + max(0.0, remaining_s - SAFETY),
        )

        # --------------------------------------------------------
        # Improve several basins.  Larger instances get more effort.
        # --------------------------------------------------------
        starts = list(initial_orders)
        if n >= 70:
            starts += [randomized_order(order_edd(instance), rng, 0.08)]
            starts += [randomized_order(order_mixed(instance), rng, 0.08)]
            starts += [randomized_order(order_mdd(instance, True), rng, 0.08)]

        per_start = 0.42 if n >= 70 else 0.50
        for initial in starts:
            if time.perf_counter() >= deadline:
                break

            local_deadline = min(deadline, time.perf_counter() + per_start)
            candidate, c = insertion_descent(instance, initial, local_deadline)

            if n >= 70 and time.perf_counter() < local_deadline:
                candidate, c = block2_descent(instance, candidate, local_deadline)

            if c < best_cost:
                receipt = submit_candidate({"order": candidate})
                if receipt.get("accepted", True):
                    best_order = list(candidate)
                    best_cost = c
                    remaining_s = receipt.get("remaining_s", TIME_LIMIT)
                    deadline = min(
                        deadline,
                        time.perf_counter() + max(0.0, remaining_s - SAFETY),
                    )

        # --------------------------------------------------------
        # Iterated large-neighborhood search.
        # --------------------------------------------------------
        current = list(best_order)
        current_cost = best_cost
        iteration = 0

        while time.perf_counter() < deadline:
            iteration += 1

            # Use the current best as the anchor, but periodically perturb it.
            if iteration % 7 == 0:
                candidate = randomized_order(best_order, rng, 0.12 if n >= 70 else 0.08)
                candidate_deadline = min(deadline, time.perf_counter() + 0.30)
                candidate, c = insertion_descent(instance, candidate, candidate_deadline)
            elif iteration % 5 == 0:
                candidate, c = block2_descent(
                    instance,
                    current,
                    min(deadline, time.perf_counter() + 0.28),
                )
                candidate, c = insertion_descent(
                    instance,
                    candidate,
                    min(deadline, time.perf_counter() + 0.22),
                )
            elif iteration % 4 == 0:
                candidate, c = swap_descent(
                    instance,
                    current,
                    min(deadline, time.perf_counter() + 0.22),
                )
            else:
                candidate = destroy_repair(
                    instance,
                    best_order,
                    rng,
                    deadline=min(deadline, time.perf_counter() + 0.32),
                )
                candidate, c = insertion_descent(
                    instance,
                    candidate,
                    min(deadline, time.perf_counter() + 0.26),
                )

            # Submit only genuine improvements.
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
                        time.perf_counter() + max(0.0, remaining_s - SAFETY),
                    )
                    continue

            # Mild acceptance of a worse diversified state so the search
            # can cross shallow local minima.  Never replaces best_order.
            if c < current_cost:
                current = list(candidate)
                current_cost = c
            else:
                delta = c - current_cost
                temperature = max(25.0, best_cost * 0.004)
                probability = math.exp(-min(40.0, delta / temperature))
                if rng.random() < 0.06 * probability:
                    current = list(candidate)
                    current_cost = c

        return {"order": best_order}
