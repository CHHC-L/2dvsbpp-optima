"""opp track: CP-SAT 2D orthogonal packing (OPP) decision for ONE fixed bin, and a fixed
multi-bin vector.

Validity-safe reductions (each preserves feasibility):
  * normal patterns: x_i in subset sums of widths of the OTHER items (<= W - w_i), same for y
    (any feasible packing can be pushed left/down until every item touches the wall or another
    item, which makes every coordinate such a sum);
  * reflection: one chosen item k that has NO identical copy is restricted to
    x_k <= (W - w_k) // 2 and y_k <= (H - h_k) // 2 (reflect first, then normalise: pushing
    left/down only decreases coordinates, then re-label identical items);
  * identical items i < j (same w,h): x_i <= x_j, and x_i == x_j -> y_i + h <= y_j
    (re-labelling identical items sorts them lexicographically by (x, y)).
  * redundant cumulatives in both directions.
A time-out is UNKNOWN, never infeasible.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402


def _cp():
    from ortools.sat.python import cp_model
    return cp_model


def opp_single(items, W, H, tl=60.0, workers=1, sym=True, normal=True, verbose=False, seed=0):
    """returns (status in FEASIBLE/INFEASIBLE/UNKNOWN, placement [(w,h,x,y)] or None, time)."""
    cp_model = _cp()
    n = len(items)
    t0 = time.time()
    if any(w > W or h > H for w, h in items) or sum(w * h for w, h in items) > W * H:
        return 'INFEASIBLE', None, 0.0
    m = cp_model.CpModel()
    ws = [w for w, h in items]; hs = [h for w, h in items]
    X = []; Y = []; xi = []; yi = []
    for i, (w, h) in enumerate(items):
        if normal:
            dx = oc.normal_patterns(ws, i, W - w)
            dy = oc.normal_patterns(hs, i, H - h)
        else:
            dx = list(range(W - w + 1)); dy = list(range(H - h + 1))
        x = m.NewIntVarFromDomain(cp_model.Domain.FromValues(dx), f'x{i}')
        y = m.NewIntVarFromDomain(cp_model.Domain.FromValues(dy), f'y{i}')
        X.append(x); Y.append(y)
        xi.append(m.NewFixedSizeIntervalVar(x, w, f'ix{i}'))
        yi.append(m.NewFixedSizeIntervalVar(y, h, f'iy{i}'))
    m.AddNoOverlap2D(xi, yi)
    m.AddCumulative(xi, hs, H)
    m.AddCumulative(yi, ws, W)
    if sym:
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault(it, []).append(i)
        uniq = [g[0] for g in groups.values() if len(g) == 1]
        if uniq:
            k = max(uniq, key=lambda i: (items[i][0] * items[i][1], i))
            m.Add(X[k] <= (W - items[k][0]) // 2)
            m.Add(Y[k] <= (H - items[k][1]) // 2)
        for g in groups.values():
            for a, b in zip(g, g[1:]):
                m.Add(X[a] <= X[b])
                eq = m.NewBoolVar('')
                m.Add(X[a] == X[b]).OnlyEnforceIf(eq)
                m.Add(X[a] < X[b]).OnlyEnforceIf(eq.Not())
                m.Add(Y[a] + items[a][1] <= Y[b]).OnlyEnforceIf(eq)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    s.parameters.log_search_progress = verbose
    st = s.Solve(m)
    dt = time.time() - t0
    if st == cp_model.INFEASIBLE:
        return 'INFEASIBLE', None, dt
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pl = [(items[i][0], items[i][1], s.Value(X[i]), s.Value(Y[i])) for i in range(n)]
        ok = oc.verify_place(items, {0: (W, H)}, [(0, w, h, x, y) for w, h, x, y in pl])
        return ('FEASIBLE' if ok else 'VERIFYFAIL'), pl, dt
    return 'UNKNOWN', None, dt


def multi_bin(items, bins_list, tl=60.0, workers=1, verbose=False, dff_pairs=None,
              geometric=True, all_nonempty=False):
    """Decide whether items fit into the fixed list of bins [(key, W, H)] (each bin used at most
    once).  geometric=False gives a pure relaxation (assignment + area + DFF capacities):
    INFEASIBLE of the relaxation is a valid infeasibility proof of the vector.
    dff_pairs: dict key -> list of (fw_table, gh_table, cap) DFF capacity rows for that bin.
    Symmetry: copies of the same (W,H) are ordered by the smallest item index they hold (valid:
    re-label identical bins)."""
    cp_model = _cp()
    t0 = time.time()
    n = len(items)
    m = cp_model.CpModel()
    a = {}
    pos = {}
    groups = {}
    for b, (key, W, H) in enumerate(bins_list):
        groups.setdefault((W, H), []).append(b)
    rank = {}
    for (W, H), bs in groups.items():
        for r, b in enumerate(bs):
            rank[b] = r
    for b, (key, W, H) in enumerate(bins_list):
        xi = []; yi = []; lits = []; idx = []
        for i, (w, h) in enumerate(items):
            if w > W or h > H:
                continue
            if i < rank[b]:
                continue  # bin copy r of a type holds only items with index >= r
            v = m.NewBoolVar(f'a{i}_{b}')
            a[i, b] = v; lits.append(v); idx.append(i)
            if geometric:
                x = m.NewIntVar(0, W - w, ''); y = m.NewIntVar(0, H - h, '')
                pos[i, b] = (x, y)
                xi.append(m.NewOptionalFixedSizeIntervalVar(x, w, v, ''))
                yi.append(m.NewOptionalFixedSizeIntervalVar(y, h, v, ''))
        m.Add(sum(items[i][0] * items[i][1] * a[i, b] for i in idx) <= W * H)
        if all_nonempty:
            m.AddBoolOr(lits) if lits else m.AddBoolOr([])
        if dff_pairs and key in dff_pairs:
            for ft, gt, cap in dff_pairs[key]:
                m.Add(sum(ft[items[i][0]] * gt[items[i][1]] * a[i, b] for i in idx) <= cap)
        if geometric and xi:
            m.AddNoOverlap2D(xi, yi)
            m.AddCumulative(xi, [items[i][1] for i in idx], H)
            m.AddCumulative(yi, [items[i][0] for i in idx], W)
    # ordering of identical bins: smallest item in copy r < smallest item in copy r+1
    for (W, H), bs in groups.items():
        for b0, b1 in zip(bs, bs[1:]):
            # if item i is in b1 then some item j < i is in b0  (lexicographic min-index order)
            for i in range(n):
                if (i, b1) in a:
                    prev = [a[j, b0] for j in range(i) if (j, b0) in a]
                    m.AddBoolOr(prev + [a[i, b1].Not()])
    for i in range(n):
        lits = [v for (j, b), v in a.items() if j == i]
        if not lits:
            return 'INFEASIBLE', None, time.time() - t0
        m.AddExactlyOne(lits)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    st = s.Solve(m)
    dt = time.time() - t0
    if st == cp_model.INFEASIBLE:
        return 'INFEASIBLE', None, dt
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if not geometric:
            return 'RELAX_FEASIBLE', None, dt
        pl = [(bins_list[b][0], items[i][0], items[i][1], s.Value(pos[i, b][0]), s.Value(pos[i, b][1]))
              for (i, b), v in a.items() if s.Value(v)]
        ok = oc.verify_place(items, {k: (W, H) for k, W, H in bins_list}, pl)
        return ('FEASIBLE' if ok else 'VERIFYFAIL'), pl, dt
    return 'UNKNOWN', None, dt
