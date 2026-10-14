"""Direct CP-SAT model for the 2DVSBPP (oriented, non-guillotine, unlimited bins per type).

Ported unchanged (except paths / CLI) from the 2026-10-13 verification probe ('vs.py').

Model
  * items sorted by non-increasing area;
  * for every bin type t: K_t = min(#items fitting in t, floor(UB / c_t)) bin 'slots' (t, k);
    u[t,k] = slot used, a[i,t,k] = item i placed in slot (t,k);
  * symmetry breaking per type: u[t,k+1] -> u[t,k], and item i may only enter slot k if i >= k
    (sort the used bins of a type by smallest item index);
  * per slot: optional intervals + AddNoOverlap2D, area cut sum(w*h*a) <= W*H*u,
    and two redundant cumulatives;
  * each item exactly once; objective sum c_t * u[t,k] with obj <= UB.
UB default = packingsolver BKS of the AREA variant; it is also valid for the weighted variant
because every bin of the benchmark has cost <= W*H (checked for all 500 instances).

Every solution returned is re-checked by `verify` (geometry, multiset, cost); an independent checker
that re-reads the raw instance file is check_solution.py.
"""
import argparse
import json
import math
import os
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402


def solve(nm, tl, workers=4, ub=None, verbose=False, area=False, return_placement=False):
    bins, items = io.read(nm, area)
    n = len(items)
    items = sorted(items, key=lambda it: -it[0] * it[1])
    if ub is None:
        ub = io.bk[nm][0]
    m = cp_model.CpModel()
    x = {}
    pos = {}
    cost_terms = []
    for t, (W, H, c) in enumerate(bins):
        fit = [i for i in range(n) if items[i][0] <= W and items[i][1] <= H]
        if not fit:
            continue
        K = min(len(fit), ub // c)
        if K == 0:
            continue
        us = []
        for k in range(K):
            u = m.NewBoolVar(f'u{t}_{k}')
            us.append(u)
            cost_terms.append(c * u)
            xi = []; yi = []; ar = []; lits = []
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
                ar.append(w * h); lits.append(a)
            m.AddNoOverlap2D(xi, yi)
            m.Add(sum(ar[j] * lits[j] for j in range(len(lits))) <= W * H * u)
            m.AddCumulative(xi, [items[i][1] for i in fit if i >= k], H)
            m.AddCumulative(yi, [items[i][0] for i in fit if i >= k], W)
        for k in range(K - 1):
            m.AddImplication(us[k + 1], us[k])
    for i in range(n):
        m.AddExactlyOne([v for (j, t, k), v in x.items() if j == i])
    obj = sum(cost_terms)
    m.Add(obj <= ub)
    m.Minimize(obj)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    t0 = time.time()
    st = s.Solve(m)
    dt = time.time() - t0
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        val = int(round(s.ObjectiveValue())); lb = int(math.ceil(s.BestObjectiveBound() - 1e-6))
        place = []
        for (i, t, k), a in x.items():
            if s.Value(a):
                place.append((t, k, items[i][0], items[i][1], s.Value(pos[i, t, k][0]), s.Value(pos[i, t, k][1])))
        ok = verify(bins, sorted(items), place, val)
        res = (s.StatusName(st) + ('' if ok else '_VERIFYFAIL'), val, lb, dt)
        return (res + (dict(bins=bins, place=place, val=val),)) if return_placement else res
    res = (s.StatusName(st), None, None, dt)
    return (res + (None,)) if return_placement else res


def verify(bins, items, place, val):
    """items: list of (w,h); place: (t,k,w,h,x,y)."""
    if sorted((p[2], p[3]) for p in place) != sorted(items):
        return False
    used = {}
    for t, k, w, h, x0, y0 in place:
        W, H, c = bins[t]
        if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            return False
        used.setdefault((t, k), []).append((x0, y0, w, h))
    for key, rs in used.items():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                xa, ya, wa, ha = rs[a]; xb, yb, wb, hb = rs[b]
                if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                    return False
    return sum(bins[t][2] for (t, k) in used) == val


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variant', choices=['weighted', 'area'], default='weighted')
    ap.add_argument('--time', type=float, default=30.0, help='time limit per instance (s)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ub', type=int, default=None, help='upper bound (default: packingsolver area BKS)')
    ap.add_argument('--save-dir', default=None, help='write solution json files here')
    ap.add_argument('names', nargs='+', help='instance names, e.g. MB_C3_11')
    a = ap.parse_args()
    area = a.variant == 'area'
    print('# name n variant bks_area bound_area status value lb time_s', flush=True)
    for nm in a.names:
        st, v, lb, dt, sol = solve(nm, a.time, workers=a.workers, ub=a.ub, area=area, return_placement=True)
        print(nm, io.n_items(nm), a.variant, io.bk[nm][0], io.bk[nm][1], st, v, lb, round(dt, 1), flush=True)
        if a.save_dir and sol is not None:
            os.makedirs(a.save_dir, exist_ok=True)
            sol.update(name=nm, variant=a.variant, status=st, lb=lb)
            json.dump(sol, open(os.path.join(a.save_dir, f'{nm}_{a.variant}.json'), 'w'))
