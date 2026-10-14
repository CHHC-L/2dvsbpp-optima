"""Sweep track (sprint 2026-10-14): improved copy of cpsat_direct.solve.

Same per-type slot model as src/cpsat_direct.py (items sorted by non-increasing area; slots
(t, k) per bin type; u[t,k+1] -> u[t,k]; item i may enter slot k only if i >= k; per-slot
NoOverlap2D + area cut + two cumulatives; obj <= UB), plus:

  * duplicate / dominated bin types are dropped (type b is dominated by type a if W_a >= W_b,
    H_a >= H_b, c_a <= c_b; replacing every b-bin by an a-bin keeps feasibility and does not
    increase cost, so the optimum is unchanged). Kept types keep their RAW file index, so the
    output packing refers to raw bin lines directly;
  * subset-area cuts on the bin counts N_t = sum_k u[t,k]: for every non-empty subset S of kept
    types, sum of areas of items that fit only in types of S <= sum_{t in S} W_t H_t N_t
    (S = all types gives the global area bound);
  * a complete solution hint built from a known packing (any packing, any variant: it is
    re-costed, mapped onto dominating kept types and re-indexed into the slot structure);
  * the final LB is CP-SAT's best bound, capped at UB, rounded up to the smallest achievable sum
    of RAW bin costs (lattice rounding; cg_bound.lattice_roundup logic re-implemented here
    to avoid importing GLOP);
  * optional early stop when a solution reaches a known external LB (that LB is NOT used in the
    model, so our own LB stays independent of other methods' bounds).

LB validity: let OPT be the optimum and UB >= OPT the value of a checked packing (or the CSV
area BKS, which is >= OPT for both variants). Every reduction above keeps at least one optimal
solution feasible in the model (dominance; K_t = min(#fitting items, UB // c_t); symmetry
breaking by sorting the used bins of a type by smallest item index; the subset-area cuts are
implied by any feasible packing), so CP-SAT's bound is <= OPT. Rounding to the cost lattice is
valid because OPT is itself a sum of raw bin costs.
"""
import itertools
import math
import os
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402


def lattice_roundup(lb, costs, cap):
    """smallest achievable sum of (integer) costs >= lb; cap = an achievable value (UB)."""
    target = int(math.ceil(lb - 1e-6))
    if target >= cap:
        return cap
    reach = bytearray(cap + 1); reach[0] = 1
    for c in sorted(set(costs)):
        for v in range(c, cap + 1):
            if reach[v - c]:
                reach[v] = 1
    for v in range(max(target, 0), cap + 1):
        if reach[v]:
            return v
    return cap


def packing_cost(bins, place):
    return sum(bins[t][2] for (t, k) in {(p[0], p[1]) for p in place})


def kept_types(bins):
    """raw indices of non-dominated bin types and map raw index -> kept dominating raw index."""
    m = len(bins)
    keep = []
    dom = {}
    for b in range(m):
        Wb, Hb, cb = bins[b]
        by = None
        for a in range(m):
            if a == b:
                continue
            Wa, Ha, ca = bins[a]
            if Wa >= Wb and Ha >= Hb and ca <= cb:
                strict = (Wa, Ha, ca) != (Wb, Hb, cb)
                if strict or a < b:  # identical types: keep the lowest index
                    by = a
                    break
        if by is None:
            keep.append(b)
        else:
            dom[b] = by
    # resolve chains to a kept type
    res = {}
    for b in range(m):
        a = b
        seen = set()
        while a in dom and a not in seen:
            seen.add(a); a = dom[a]
        res[b] = a
    return keep, res


class _StopAt(cp_model.CpSolverSolutionCallback):
    def __init__(self, target):
        super().__init__(); self.target = target

    def on_solution_callback(self):
        if self.target is not None and self.ObjectiveValue() <= self.target + 1e-9:
            self.StopSearch()


def solve(nm, variant, tl, workers=3, ub=None, hint_place=None, stop_at=None, verbose=False,
          area_cuts=True, dominance=True, seed=None):
    """returns dict(status, val, lb_raw, lb, place, time, hinted, hint_cost, ub_used, ...)."""
    area = variant == 'area'
    bins, items_raw = io.read(nm, area)
    n = len(items_raw)
    order = sorted(range(n), key=lambda i: -items_raw[i][0] * items_raw[i][1])
    items = [items_raw[i] for i in order]
    if ub is None:
        ub = io.bk[nm][0]
    if dominance:
        keep, tmap = kept_types(bins)
    else:
        keep, tmap = list(range(len(bins))), {t: t for t in range(len(bins))}
    m = cp_model.CpModel()
    x = {}; pos = {}; U = {}
    cost_terms = []
    fitsets = [frozenset(t for t in keep if items[i][0] <= bins[t][0] and items[i][1] <= bins[t][1])
               for i in range(n)]
    for t in keep:
        W, H, c = bins[t]
        fit = [i for i in range(n) if t in fitsets[i]]
        if not fit:
            continue
        K = min(len(fit), ub // c)
        if K == 0:
            continue
        us = []
        for k in range(K):
            u = m.NewBoolVar(f'u{t}_{k}')
            us.append(u); U[t, k] = u
            cost_terms.append(c * u)
            xi = []; yi = []; ar = []; lits = []; hs = []; ws = []
            for i in fit:
                if i < k:
                    continue
                w, h = items[i]
                a = m.NewBoolVar(f'a{i}_{t}_{k}')
                x[i, t, k] = a
                m.AddImplication(a, u)
                xv = m.NewIntVar(0, W - w, ''); yv = m.NewIntVar(0, H - h, '')
                pos[i, t, k] = (xv, yv)
                xi.append(m.NewOptionalFixedSizeIntervalVar(xv, w, a, ''))
                yi.append(m.NewOptionalFixedSizeIntervalVar(yv, h, a, ''))
                ar.append(w * h); lits.append(a); hs.append(h); ws.append(w)
            m.AddNoOverlap2D(xi, yi)
            m.Add(sum(ar[j] * lits[j] for j in range(len(lits))) <= W * H * u)
            m.AddCumulative(xi, hs, H)
            m.AddCumulative(yi, ws, W)
        for k in range(K - 1):
            m.AddImplication(us[k + 1], us[k])
    for i in range(n):
        lits = [v for (j, t, k), v in x.items() if j == i]
        if not lits:
            return dict(status='NOFIT', val=None, lb_raw=None, lb=None, place=None, time=0.0)
        m.AddExactlyOne(lits)
    if area_cuts:
        types_used = sorted({t for (t, k) in U})
        for r in range(1, len(types_used) + 1):
            for S in itertools.combinations(types_used, r):
                Sset = set(S)
                need = sum(items[i][0] * items[i][1] for i in range(n) if fitsets[i] <= Sset)
                if need <= 0:
                    continue
                m.Add(sum(bins[t][0] * bins[t][1] * U[t, k] for (t, k) in U if t in Sset) >= need)
    obj = sum(cost_terms)
    m.Add(obj <= ub)
    m.Minimize(obj)

    hinted = False; hint_cost = None
    if hint_place:
        hp = _map_hint(hint_place, items, tmap, bins, U, x)
        if hp is not None:
            hint_cost = packing_cost(bins, [(t, k) for (t, k, *_r) in hp])  # cost of mapped packing
            if hint_cost <= ub:
                used = {(t, k) for (t, k, i, xx, yy) in hp}
                for key, u in U.items():
                    m.AddHint(u, 1 if key in used else 0)
                assign = {(i, t, k): (xx, yy) for (t, k, i, xx, yy) in hp}
                for key, a in x.items():
                    if key in assign:
                        m.AddHint(a, 1)
                        m.AddHint(pos[key][0], assign[key][0]); m.AddHint(pos[key][1], assign[key][1])
                    else:
                        m.AddHint(a, 0)
                hinted = True

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    if seed is not None:
        s.parameters.random_seed = seed
    cb = _StopAt(stop_at)
    t0 = time.time()
    st = s.Solve(m, cb)
    dt = time.time() - t0
    out = dict(status=s.StatusName(st), time=dt, hinted=hinted, hint_cost=hint_cost, ub_used=ub,
               kept=keep, val=None, lb_raw=None, lb=None, place=None)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        val = int(round(s.ObjectiveValue()))
        lbr = int(math.ceil(s.BestObjectiveBound() - 1e-6))
        if st == cp_model.OPTIMAL and not cb_stopped(s, val, stop_at):
            lbr = val
        lbr = min(lbr, val)
        place = []
        for (i, t, k), a in x.items():
            if s.Value(a):
                place.append([t, k, items[i][0], items[i][1], s.Value(pos[i, t, k][0]), s.Value(pos[i, t, k][1])])
        out.update(val=val, lb_raw=lbr, lb=lattice_roundup(lbr, [b[2] for b in bins], val), place=place)
    elif st == cp_model.UNKNOWN:
        lbr = int(math.ceil(s.BestObjectiveBound() - 1e-6)) if math.isfinite(s.BestObjectiveBound()) else 0
        lbr = min(lbr, ub)
        out.update(lb_raw=lbr, lb=lattice_roundup(lbr, [b[2] for b in bins], ub))
    # INFEASIBLE: would mean OPT > ub, i.e. the given UB is not achievable -> flag, no LB claimed
    return out


def cb_stopped(s, val, stop_at):
    """CP-SAT reports OPTIMAL only when the bound meets the objective; StopSearch gives FEASIBLE.
    Kept as a hook: never treat a callback stop as an optimality proof."""
    return False


def _map_hint(place, items, tmap, bins, U, x):
    """place: [[t_raw, k, w, h, x, y], ...] -> [(t_kept, slot, item_index, x, y)] or None."""
    pool = {}
    for i, (w, h) in enumerate(items):
        pool.setdefault((w, h), []).append(i)
    for v in pool.values():
        v.sort(reverse=True)  # pop() gives smallest index first
    groups = {}
    for t, k, w, h, xx, yy in place:
        groups.setdefault((t, k), []).append((w, h, xx, yy))
    bins_by_type = {}
    for (t, k), rs in groups.items():
        tk = tmap.get(t, t)
        if tk not in {tt for (tt, kk) in U}:
            return None
        lst = []
        for w, h, xx, yy in rs:
            if not pool.get((w, h)):
                return None
            lst.append((pool[(w, h)].pop(), xx, yy))
        bins_by_type.setdefault(tk, []).append(lst)
    if any(pool[k] for k in pool):
        return None
    out = []
    for tk, blist in bins_by_type.items():
        blist.sort(key=lambda L: min(i for i, _, _ in L))
        for slot, L in enumerate(blist):
            if (tk, slot) not in U:
                return None
            for i, xx, yy in L:
                if (i, tk, slot) not in x:
                    return None
                out.append((tk, slot, i, xx, yy))
    return out
