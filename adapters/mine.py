"""
Autoclave Queue solver  ->  adapters/mine.py

Strategy (tuned for the anytime scoring at 5% / 20% / 50% / 100% of 5 s):

1. Submit EDD immediately so a valid plan always exists.
2. Build several release/setup-aware dispatch schedules and keep the best.
3. Run simulated annealing in four time slices that END at the scoring
   checkpoints.  Each slice restarts from the incumbent and cools to ~0, so
   the best-so-far at every checkpoint is a freshly polished local optimum.
4. Moves: block insertion (1-3 batches), swap.  Every move only re-simulates
   the changed window plus the tail, and the tail is cut short when
   (a) a release date re-synchronises the schedule with the old one, or
   (b) a lower bound already exceeds the acceptance threshold.
"""
import math
import random
import time
from itertools import permutations

from adapter import Solver

try:
    from adapter import cost_of
except Exception:  # pragma: no cover
    cost_of = None

# ---- tunables ---------------------------------------------------------------
TIME_LIMIT = 4.6                    # stop searching here (budget is 5.0 s)
SEG_ENDS = (0.22, 0.95, 2.45)       # slice ends just before the 5%/20%/50% marks
SEG_TEMPS = ((0.02, 0.002), (0.03, 0.003), (0.03, 0.003), (0.05, 0.002))
EMIT_GAP = 0.05                     # min seconds between intermediate submits
NEG = -1e18


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        return _Run(instance, submit_candidate).run()


class _Run:
    def __init__(self, inst, submit):
        self.t0 = time.perf_counter()
        self.inst = inst
        self.submit = submit
        n = self.n = int(inst.size)
        fam = [int(x) for x in inst.fam]
        self.proc = [inst.proc[b] for b in range(n)]
        self.rel = [inst.release[b] for b in range(n)]
        self.due = [inst.due[b] for b in range(n)]
        self.w = [inst.weight[b] for b in range(n)]
        self.J = [(self.rel[b], self.proc[b], self.due[b], self.w[b]) for b in range(n)]
        su = inst.setup
        # ST[a][b] = changeover from batch a to batch b; row n = "nothing yet" (0)
        self.ST = [[su[fam[a]][fam[b]] for b in range(n)] for a in range(n)] + [[0] * n]
        self.pbar = max(sum(self.proc) / n, 1e-9)
        nz = [su[f][g] for f in range(len(su)) for g in range(len(su)) if f != g]
        sb = (sum(nz) / len(nz)) if nz else 0
        self.sbar = sb if sb > 0 else 1.0
        self.rng = random.Random(20240601)
        self.W = max(3, n // 6)
        self.last_emit = -1.0
        self.pending = False
        self.best_cost = None
        self.best_order = None

    # ---- helpers -------------------------------------------------------------
    def now(self):
        return time.perf_counter() - self.t0

    def emit(self, order):
        try:
            self.submit_candidate_safe(order)
        except Exception:
            pass

    def submit_candidate_safe(self, order):
        self.submit({"order": [int(b) for b in order]})

    def full_cost(self, order):
        J = self.J; ST = self.ST
        t = NEG; prev = self.n; td = 0
        for b in order:
            rb, pb, db, wb = J[b]
            s = t + ST[prev][b]
            if s < rb:
                s = rb
            t = s + pb
            x = t - db
            if x > 0:
                td += wb * x
            prev = b
        return 4 * t + td

    def load(self, order):
        """Make `order` the current solution and rebuild the incremental caches."""
        n = self.n; J = self.J; ST = self.ST
        self.cur = list(order)
        fin = [0] * n; T = [0] * (n + 1); Prem = [0] * (n + 1)
        t = NEG; prev = n; td = 0
        for p, b in enumerate(self.cur):
            rb, pb, db, wb = J[b]
            s = t + ST[prev][b]
            if s < rb:
                s = rb
            t = s + pb
            x = t - db
            if x > 0:
                td += wb * x
            fin[p] = t; T[p + 1] = td; prev = b
        for p in range(n - 1, -1, -1):
            Prem[p] = Prem[p + 1] + J[self.cur[p]][1]
        self.fin = fin; self.T = T; self.Prem = Prem
        self.cost = 4 * t + td

    def consider(self, order, cost=None):
        if cost is None:
            cost = self.full_cost(order)
        if self.best_cost is None or cost < self.best_cost:
            self.best_cost = cost
            self.best_order = list(order)
            self.pending = True

    # ---- constructive heuristics ---------------------------------------------
    def construct(self, kind, k1=1.0, k2=1.0):
        n = self.n; J = self.J; ST = self.ST
        left = list(range(n))
        t = 0; prev = n; out = []
        pbar = self.pbar; sbar = self.sbar
        exp = math.exp
        while left:
            row = ST[prev]
            best_b = -1; best_v = None
            for b in left:
                rb, pb, db, wb = J[b]
                s = t + row[b]
                if s < rb:
                    s = rb
                c = s + pb
                if kind == 0:      # ATCS-style index with release/setup awareness
                    sl = db - c
                    if sl < 0:
                        sl = 0
                    v = -(wb / pb) * exp(-sl / (k1 * pbar)) * exp(-(s - t) / (k2 * sbar))
                elif kind == 1:    # earliest completion
                    v = c + db * 1e-6
                else:              # cheapest cost increment per unit of work
                    late = c - db
                    v = (4 * (c - t) + (wb * late if late > 0 else 0)) / pb
                if best_v is None or v < best_v:
                    best_v = v; best_b = b
            left.remove(best_b)
            out.append(best_b)
            s = t + row[best_b]
            if s < J[best_b][0]:
                s = J[best_b][0]
            t = s + J[best_b][1]
            prev = best_b
        return out

    # ---- simulated annealing slice -------------------------------------------
    def anneal(self, t_end, f0, f1):
        n = self.n
        J = self.J; ST = self.ST
        order = self.cur; fin = self.fin; T = self.T; Prem = self.Prem
        cost = self.cost
        best_cost = self.best_cost; best_order = self.best_order
        pending = self.pending
        rand = self.rng.random
        log = math.log; exp = math.exp
        clock = time.perf_counter
        W = self.W; W2 = 2 * W + 1
        n1 = n - 1
        t0 = self.t0
        temp0 = f0 * self.dbar; temp1 = f1 * self.dbar
        if temp0 <= 0:
            temp0 = temp1 = 1e-9
        rl = log(temp1 / temp0)
        start = clock() - t0
        span = max(t_end - start, 1e-6)
        temp = temp0
        it = 0
        while True:
            it += 1
            if not (it & 15):
                now = clock() - t0
                if now >= t_end:
                    break
                temp = temp0 * exp(rl * (now - start) / span)
                if pending and now - self.last_emit >= EMIT_GAP:
                    self.emit(best_order)
                    self.last_emit = now
                    pending = False
            r = rand()
            if r < 0.25:                                   # swap
                i = int(rand() * n)
                if rand() < 0.7:
                    j = i + int(rand() * W2) - W
                    if j < 0 or j >= n:
                        continue
                else:
                    j = int(rand() * n)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                lo = i; hi = j
                seg = [order[j]]
                if j - i > 1:
                    seg.extend(order[i + 1:j])
                seg.append(order[i])
            else:                                          # block insertion
                L = 1 if r < 0.75 else (2 if r < 0.90 else 3)
                m = n - L
                i = int(rand() * (m + 1))
                if rand() < 0.7:
                    j = i + int(rand() * W2) - W
                    if j < 0 or j > m:
                        continue
                else:
                    j = int(rand() * (m + 1))
                if j == i:
                    continue
                if j > i:
                    lo = i; hi = j + L - 1
                    seg = order[i + L:j + L]
                    seg.extend(order[i:i + L])
                else:
                    lo = j; hi = i + L - 1
                    seg = order[i:i + L]
                    seg.extend(order[j:i])
            # ---- evaluate (suffix only) ----
            thr = cost - temp * log(1.0 - rand())
            if lo:
                t = fin[lo - 1]; prev = order[lo - 1]
            else:
                t = NEG; prev = n
            td = T[lo]
            for b in seg:
                rb, pb, db, wb = J[b]
                s = t + ST[prev][b]
                if s < rb:
                    s = rb
                t = s + pb
                x = t - db
                if x > 0:
                    td += wb * x
                prev = b
            if td + 4 * (t + Prem[hi + 1]) > thr:
                continue
            ok = True
            new_cost = 0
            for p in range(hi + 1, n):
                b = order[p]
                rb, pb, db, wb = J[b]
                s = t + ST[prev][b]
                if s < rb:
                    s = rb
                t = s + pb
                x = t - db
                if x > 0:
                    td += wb * x
                if t == fin[p]:                            # re-synchronised
                    new_cost = 4 * fin[n1] + td + T[n] - T[p + 1]
                    break
                if td + 4 * (t + Prem[p + 1]) > thr:
                    ok = False
                    break
                prev = b
            else:
                new_cost = 4 * t + td
            if not ok or new_cost > thr:
                continue
            # ---- accept: apply move and refresh caches from lo ----
            order[lo:hi + 1] = seg
            if lo:
                t = fin[lo - 1]; prev = order[lo - 1]
            else:
                t = NEG; prev = n
            td = T[lo]
            for p in range(lo, n):
                b = order[p]
                rb, pb, db, wb = J[b]
                s = t + ST[prev][b]
                if s < rb:
                    s = rb
                t = s + pb
                x = t - db
                if x > 0:
                    td += wb * x
                fin[p] = t; T[p + 1] = td; prev = b
            for p in range(hi, lo, -1):
                Prem[p] = Prem[p + 1] + J[order[p]][1]
            cost = 4 * t + td
            if cost < best_cost:
                best_cost = cost
                best_order = order[:]
                pending = True
        self.cost = cost
        self.best_cost = best_cost; self.best_order = best_order
        self.pending = pending
        self.iters = getattr(self, "iters", 0) + it

    def calibrate(self):
        """Typical uphill move size, used to scale temperatures."""
        n = self.n
        base = self.best_order
        c0 = self.best_cost
        rng = self.rng
        ups = []
        for _ in range(250):
            i = rng.randrange(n); j = rng.randrange(n)
            if i == j:
                continue
            o = base[:]
            o.insert(j, o.pop(i))
            d = self.full_cost(o) - c0
            if d > 0:
                ups.append(d)
        if not ups:
            return max(1.0, abs(c0) * 0.001)
        ups.sort()
        return ups[len(ups) // 2]

    # ---- main ------------------------------------------------------------------
    def run(self):
        n = self.n
        due = self.due; rel = self.rel
        edd = sorted(range(n), key=lambda b: (due[b], rel[b], b))
        self.emit(edd)
        self.consider(edd)
        self.pending = False
        if n <= 1:
            return {"order": edd}
        if n <= 7:
            best = min(permutations(range(n)), key=self.full_cost)
            self.emit(best)
            return {"order": list(best)}

        self.consider(sorted(range(n), key=lambda b: (rel[b], due[b], b)))
        cons = [self.construct(1), self.construct(2)]
        for k1 in (1.0, 3.0, 10.0, 30.0):
            for k2 in (0.3, 1.0, 3.0, 10.0):
                if self.now() > SEG_ENDS[0] * 0.45:
                    break
                cons.append(self.construct(0, k1, k2))
        for o in cons:
            self.consider(o)
        self.emit(self.best_order)
        self.last_emit = self.now()
        self.pending = False

        self.dbar = self.calibrate()
        ends = list(SEG_ENDS) + [TIME_LIMIT]
        for end, (f0, f1) in zip(ends, SEG_TEMPS):
            end = min(end, TIME_LIMIT)
            if self.now() >= end - 0.01:
                continue
            self.load(self.best_order)
            self.anneal(end, f0, f1)
            if self.pending:
                self.emit(self.best_order)
                self.last_emit = self.now()
                self.pending = False

        best = self.best_order
        if cost_of is not None:
            try:
                if cost_of(self.inst, edd) < cost_of(self.inst, best):
                    best = edd
                    self.emit(best)
            except Exception:
                pass
        return {"order": [int(b) for b in best]}