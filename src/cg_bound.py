"""Dantzig-Wolfe / column-generation lower bound for the 2DVSBPP (oriented, non-guillotine).

Ported unchanged (except paths / CLI) from the 2026-10-13 verification probe ('cg.py').

  * master: set-covering LP  min sum c_p lambda_p  s.t. sum_{p ni i} lambda_p >= 1   (GLOP);
  * pricing: per bin type an exact 2D knapsack (max sum of duals over 2D-feasible subsets),
    solved by CP-SAT with no_overlap_2d (+ area and cumulative redundancy), duals scaled by SCALE;
  * Farley bound theta * sum(duals) when a pricing problem is not solved to optimality
    (theta = min_t c_t / UB(price_t)), so the bound is valid at every iteration;
  * the LP bound is rounded up to the smallest achievable sum of bin costs (subset-sum lattice),
    which matters because there are only <= 5 distinct costs;
  * a restricted-master IP (CP-SAT over the generated columns) gives a heuristic UB.

NOTE: highspy and ortools conflicted when loaded in the same process during the probe, hence GLOP.
Pricing tolerance: duals are floored to multiples of 1/SCALE; the price upper bound adds
len(cand)/SCALE to stay valid.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402

SCALE = 10000


def solve_lp(cols, costs, n):
    lp = pywraplp.Solver.CreateSolver('GLOP')
    lam = [lp.NumVar(0, lp.infinity(), '') for _ in cols]
    rows = [lp.Constraint(1, lp.infinity()) for i in range(n)]
    for p, S in enumerate(cols):
        for i in S:
            rows[i].SetCoefficient(lam[p], 1)
    obj = lp.Objective()
    for p, c in enumerate(costs):
        obj.SetCoefficient(lam[p], float(c))
    obj.SetMinimization()
    st = lp.Solve()
    assert st == pywraplp.Solver.OPTIMAL
    duals = np.array([max(0.0, r.dual_value()) for r in rows])
    return obj.Value(), duals, np.array([v.solution_value() for v in lam])


def price(items, W, H, duals, tl, max_cols=5):
    """max sum duals over 2D-feasible subsets. returns list of (value, set), best UB on price"""
    cand = [i for i in range(len(items)) if items[i][0] <= W and items[i][1] <= H and duals[i] > 1e-9]
    if not cand:
        return [], 0.0
    m = cp_model.CpModel()
    z = {}; xi = []; yi = []
    for i in cand:
        w, h = items[i]
        z[i] = m.NewBoolVar('')
        xv = m.NewIntVar(0, W - w, ''); yv = m.NewIntVar(0, H - h, '')
        xi.append(m.NewOptionalFixedSizeIntervalVar(xv, w, z[i], ''))
        yi.append(m.NewOptionalFixedSizeIntervalVar(yv, h, z[i], ''))
    m.AddNoOverlap2D(xi, yi)
    m.Add(sum(items[i][0] * items[i][1] * z[i] for i in cand) <= W * H)
    m.AddCumulative(xi, [items[i][1] for i in cand], H)
    m.AddCumulative(yi, [items[i][0] for i in cand], W)
    wts = {i: int(math.floor(duals[i] * SCALE)) for i in cand}
    m.Maximize(sum(wts[i] * z[i] for i in cand))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 4
    st = s.Solve(m)
    out = []
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        S = frozenset(i for i in cand if s.Value(z[i]))
        out.append((sum(duals[i] for i in S), S))
        # upper bound on true price: floor-scaled weights underestimate by < 1/SCALE per item
        ub = (s.BestObjectiveBound() + len(cand)) / SCALE
    else:
        ub = sum(duals[i] for i in cand)
    return out, ub


def lattice_roundup(lb, costs, cap):
    """smallest achievable sum of costs >= lb (costs integer), cap = an achievable value (UB)"""
    target = int(math.ceil(lb - 1e-6))
    if target >= cap:
        return cap
    reach = np.zeros(cap + 1, dtype=bool); reach[0] = True
    for c in sorted(set(costs)):
        for v in range(c, cap + 1):
            if reach[v - c]:
                reach[v] = True
    for v in range(target, cap + 1):
        if reach[v]:
            return v
    return cap


def cg(nm, area, ub, tl_total=600, tl_price=10):
    bins, items = io.read(nm, area)
    n = len(items)
    t0 = time.time()
    cols = []; costs = []; types = []
    for i, (w, h) in enumerate(items):
        best = min((c, t) for t, (W, H, c) in enumerate(bins) if w <= W and h <= H)
        cols.append(frozenset([i])); costs.append(best[0]); types.append(best[1])
    seen = set((t, S) for t, S in zip(types, cols))
    best_lb = 0.0; it = 0
    while time.time() - t0 < tl_total:
        it += 1
        z, duals, _ = solve_lp(cols, costs, n)
        added = 0; theta = 1.0
        for t, (W, H, c) in enumerate(bins):
            out, pub = price(items, W, H, duals, tl_price)
            if pub > c + 1e-9:
                theta = min(theta, c / pub)
            for val, S in out:
                if val > c + 1e-7 and (t, S) not in seen:
                    seen.add((t, S)); cols.append(S); costs.append(c); types.append(t); added += 1
        lb = theta * float(np.sum(duals))
        best_lb = max(best_lb, lb)
        if added == 0:
            break
    # restricted master IP heuristic
    m = cp_model.CpModel(); lam = [m.NewBoolVar('') for _ in cols]
    for i in range(n):
        m.AddBoolOr([lam[p] for p, S in enumerate(cols) if i in S])
    m.Minimize(sum(int(c) * l for c, l in zip(costs, lam)))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = 60; s.parameters.num_workers = 4
    st = s.Solve(m)
    rmp = int(s.ObjectiveValue()) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    rl = lattice_roundup(best_lb, [c for _, _, c in bins], min(ub, rmp or ub))
    return dict(rmp_ub=rmp, name=nm, area=area, lp=z, lb=best_lb, lb_round=rl, ub=ub, iters=it,
                cols=len(cols), time=time.time() - t0, converged=(added == 0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variant', choices=['weighted', 'area'], default='weighted')
    ap.add_argument('--time', type=float, default=600.0, help='total CG time limit per instance (s)')
    ap.add_argument('--price-time', type=float, default=10.0, help='time limit per pricing problem (s)')
    ap.add_argument('--ub', type=int, default=None, help='upper bound (default: packingsolver area BKS)')
    ap.add_argument('names', nargs='+')
    a = ap.parse_args()
    for nm in a.names:
        r = cg(nm, a.variant == 'area', a.ub if a.ub is not None else io.bk[nm][0],
               tl_total=a.time, tl_price=a.price_time)
        print(r, flush=True)
