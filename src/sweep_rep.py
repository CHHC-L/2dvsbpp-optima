"""Sweep track (sprint 2026-10-14): 'representative' generic-slot CP-SAT model for the 2DVSBPP.

Items sorted by non-increasing area (index 0 = largest). There are n potential bins; bin k can
only be opened by item k, which is then the smallest-index item in it (o[k] = a[k,k]).
Item i may join bin k < i only if bin k is open. Bin k picks exactly one (kept) bin type among
those that fit item k (u[k,t]); its size (Wk, Hk) follows. Per bin: NoOverlap2D of optional
intervals with x + w <= Wk, y + h <= Hk, area cut, two cumulatives with variable capacity.
Same extras as sweep_model.py: dominated / duplicate bin types dropped, per-type count limits
floor(UB / c_t), subset-area cuts on the counts N_t, solution hint, lattice-rounded LB.

Validity of the symmetry breaking: in any packing, order the bins by their smallest item index;
re-index each bin by that index k. Then item k is in bin k and every other item i of bin k has
i > k, which is exactly the model's structure, so every packing (in particular an optimal one,
re-typed onto non-dominated types) is representable.
"""
import itertools
import math
import os
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402
from sweep_model import lattice_roundup, kept_types, _StopAt  # noqa: E402


def solve(nm, variant, tl, workers=3, ub=None, hint_place=None, stop_at=None, verbose=False,
          area_cuts=True, seed=None):
    area = variant == 'area'
    bins, items_raw = io.read(nm, area)
    n = len(items_raw)
    items = sorted(items_raw, key=lambda it: -it[0] * it[1])
    if ub is None:
        ub = io.bk[nm][0]
    keep, tmap = kept_types(bins)
    fits = [[t for t in keep if items[i][0] <= bins[t][0] and items[i][1] <= bins[t][1]] for i in range(n)]
    m = cp_model.CpModel()
    a = {}; u = {}; pos = {}
    o = []
    Wk = []; Hk = []
    for k in range(n):
        ts = [t for t in fits[k] if bins[t][2] <= ub]
        ok = m.NewBoolVar(f'o{k}'); o.append(ok)
        a[k, k] = ok
        for t in ts:
            u[k, t] = m.NewBoolVar(f'u{k}_{t}')
        m.Add(sum(u[k, t] for t in ts) == ok)
        Wmax = max([bins[t][0] for t in ts], default=0); Hmax = max([bins[t][1] for t in ts], default=0)
        wk = m.NewIntVar(0, Wmax, f'W{k}'); hk = m.NewIntVar(0, Hmax, f'H{k}')
        m.Add(wk == sum(bins[t][0] * u[k, t] for t in ts))
        m.Add(hk == sum(bins[t][1] * u[k, t] for t in ts))
        Wk.append(wk); Hk.append(hk)
    for i in range(n):
        for k in range(i):
            tsk = [t for t in fits[k] if (k, t) in u]
            okt = [t for t in tsk if t in fits[i]]
            if not okt:
                continue
            v = m.NewBoolVar(f'a{i}_{k}')
            a[i, k] = v
            m.AddImplication(v, o[k])
            for t in tsk:
                if t not in fits[i]:
                    m.AddImplication(v, u[k, t].Not())
        m.AddExactlyOne([a[i, k] for k in range(i + 1) if (i, k) in a])
    for k in range(n):
        xi = []; yi = []; ar = []; lits = []; hs = []; ws = []
        Wmax = max([bins[t][0] for t in fits[k] if (k, t) in u], default=0)
        Hmax = max([bins[t][1] for t in fits[k] if (k, t) in u], default=0)
        for i in range(k, n):
            if (i, k) not in a:
                continue
            w, h = items[i]
            lit = a[i, k]
            xv = m.NewIntVar(0, max(0, Wmax - w), ''); yv = m.NewIntVar(0, max(0, Hmax - h), '')
            m.Add(xv + w <= Wk[k]).OnlyEnforceIf(lit)
            m.Add(yv + h <= Hk[k]).OnlyEnforceIf(lit)
            pos[i, k] = (xv, yv)
            xi.append(m.NewOptionalFixedSizeIntervalVar(xv, w, lit, ''))
            yi.append(m.NewOptionalFixedSizeIntervalVar(yv, h, lit, ''))
            ar.append(w * h); lits.append(lit); hs.append(h); ws.append(w)
        if len(xi) > 1:
            m.AddNoOverlap2D(xi, yi)
            m.AddCumulative(xi, hs, Hk[k])
            m.AddCumulative(yi, ws, Wk[k])
        m.Add(sum(ar[j] * lits[j] for j in range(len(lits))) <= sum(bins[t][0] * bins[t][1] * u[k, t] for t in fits[k] if (k, t) in u))
    types_used = sorted({t for (k, t) in u})
    N = {t: sum(u[k, tt] for (k, tt) in u if tt == t) for t in types_used}
    for t in types_used:
        m.Add(N[t] <= ub // bins[t][2])
    if area_cuts:
        for r in range(1, len(types_used) + 1):
            for S in itertools.combinations(types_used, r):
                Sset = set(S)
                need = sum(items[i][0] * items[i][1] for i in range(n) if set(fits[i]) <= Sset)
                if need > 0:
                    m.Add(sum(bins[t][0] * bins[t][1] * N[t] for t in S) >= need)
    obj = sum(bins[t][2] * u[k, t] for (k, t) in u)
    m.Add(obj <= ub)
    m.Minimize(obj)

    hinted = False; hint_cost = None
    if hint_place:
        hp = _map_hint(hint_place, items, tmap, u, a)
        if hp is not None:
            hint_cost = sum(bins[t][2] for (k, t) in {(k, t) for (k, t, i, xx, yy) in hp})
            if hint_cost <= ub:
                opened = {k: t for (k, t, i, xx, yy) in hp}
                for (k, t), v in u.items():
                    m.AddHint(v, 1 if opened.get(k) == t else 0)
                for k in range(n):
                    if k not in opened:
                        m.AddHint(Wk[k], 0); m.AddHint(Hk[k], 0)
                    else:
                        m.AddHint(Wk[k], bins[opened[k]][0]); m.AddHint(Hk[k], bins[opened[k]][1])
                asg = {(i, k): (xx, yy) for (k, t, i, xx, yy) in hp}
                for key, v in a.items():
                    if key in asg:
                        m.AddHint(v, 1)
                        if key in pos:
                            m.AddHint(pos[key][0], asg[key][0]); m.AddHint(pos[key][1], asg[key][1])
                    else:
                        m.AddHint(v, 0)
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
    costs = [b[2] for b in bins]
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        val = int(round(s.ObjectiveValue()))
        lbr = val if st == cp_model.OPTIMAL else int(math.ceil(s.BestObjectiveBound() - 1e-6))
        lbr = min(lbr, val)
        typ = {k: t for (k, t), v in u.items() if s.Value(v)}
        place = []
        for (i, k), v in a.items():
            if s.Value(v):
                xx, yy = (s.Value(pos[i, k][0]), s.Value(pos[i, k][1])) if (i, k) in pos else (0, 0)
                place.append([typ[k], k, items[i][0], items[i][1], xx, yy])
        out.update(val=val, lb_raw=lbr, lb=lattice_roundup(lbr, costs, val), place=place)
    elif st == cp_model.UNKNOWN:
        bd = s.BestObjectiveBound()
        lbr = min(int(math.ceil(bd - 1e-6)) if math.isfinite(bd) else 0, ub)
        out.update(lb_raw=lbr, lb=lattice_roundup(lbr, costs, ub))
    return out


def _map_hint(place, items, tmap, u, a):
    pool = {}
    for i, (w, h) in enumerate(items):
        pool.setdefault((w, h), []).append(i)
    for v in pool.values():
        v.sort(reverse=True)
    groups = {}
    for t, k, w, h, xx, yy in place:
        groups.setdefault((t, k), []).append((w, h, xx, yy))
    # assign the largest items first so that the per-bin smallest index is well defined
    out = []
    for (t, k), rs in groups.items():
        tk = tmap.get(t, t)
        L = []
        for w, h, xx, yy in rs:
            if not pool.get((w, h)):
                return None
            L.append((pool[(w, h)].pop(), xx, yy))
        rep = min(i for i, _, _ in L)
        if (rep, tk) not in u:
            return None
        for i, xx, yy in L:
            if (i, rep) not in a:
                return None
            out.append((rep, tk, i, xx, yy))
    if any(pool[k] for k in pool):
        return None
    return out
