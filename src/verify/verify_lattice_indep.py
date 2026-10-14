"""Independent re-implementation of the cost-lattice LB argument (verify_lattice track).

Shares NO code with src/lattice_exact.py or src/vsbpp_io.py (own parser of the raw file).
Deliberately weaker / simpler than lattice_exact so that its bugs cannot be shared:
  * ALL raw bin lines are enumerated (no dominance reduction, no dedup);
  * empty bins are allowed (no non-empty requirement); n_t <= n;
  * filters: every item fits some chosen line; total area; Hall area over subsets of chosen lines
    (items that fit only into lines of S need area <= capacity of S) -- each a necessary condition;
  * per vector: CP-SAT with item->bin assignment, optional intervals + AddNoOverlap2D per bin,
    positions restricted to normal patterns (own subset-sum code), two redundant cumulatives, area
    knapsack, tall/wide cuts (items with h>H/2 have total width <= W, symmetric), pairwise
    conflicts (w_i+w_j>W and h_i+h_j>H), and as the only symmetry breaking:
    copies of the same line relabelled by smallest item index with empty copies last.
  * NOTE: items are in RAW file order (not sorted); the symmetry argument does not need sorting.
  * a vector is refuted only by the filters or CP-SAT INFEASIBLE; UNKNOWN stops the scan.
Result: LB = cost of the first vector (in cost order) not refuted; if every vector with cost < V
is refuted, OPT >= V.

usage: python src/verify/verify_lattice_indep.py <name> <weighted|area> <target_V> <total_s> <vec_s> [workers] [lo]
  proves (if successful) that no packing of cost < target_V exists.
"""
import sys
import time

from ortools.sat.python import cp_model

import os
# NONEMPTY=1: every bin of the vector must be used. Valid because ALL vectors with cost < target are
# enumerated (raw lines): dropping the empty bins of a cheaper packing gives another cheaper packing
# whose vector is also enumerated (all costs > 0).
NONEMPTY = os.environ.get('NONEMPTY', '0') == '1'
P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_raw(nm, variant):
    L = open(f'{P}/data/instances/{nm}.txt').read().split('\n')
    n, m, _ = map(int, L[0].split())
    bins = []
    for t in range(m):
        W, H, _one, c = map(int, L[1 + t].split())
        bins.append((W, H, W * H if variant == 'area' else c))
    items = []
    for i in range(n):
        _idx, w, h, _d = map(int, L[1 + m + i].split())
        items.append((w, h))
    if os.environ.get('SORTITEMS', '0') == '1':  # any fixed order is valid for the min-index symmetry
        items.sort(key=lambda it: -it[0] * it[1])
    return bins, items


def vectors(bins, n, vmax):
    out = []
    T = len(bins)

    def rec(t, cost, cnt, vec):
        if t == T:
            if cnt > 0:
                out.append((cost, tuple(vec)))
            return
        k = 0
        while cost + k * bins[t][2] <= vmax and cnt + k <= n:
            vec.append(k); rec(t + 1, cost + k * bins[t][2], cnt + k, vec); vec.pop(); k += 1
    rec(0, 0, 0, [])
    out.sort()
    return out


def filt(bins, items, vec):
    used = [t for t in range(len(bins)) if vec[t]]
    fs = []
    for w, h in items:
        f = frozenset(t for t in used if w <= bins[t][0] and h <= bins[t][1])
        if not f:
            return 'nofit'
        fs.append(f)
    ar = [w * h for w, h in items]
    for mask in range(1, 1 << len(used)):
        S = frozenset(used[b] for b in range(len(used)) if mask >> b & 1)
        if sum(ar[i] for i in range(len(items)) if fs[i] <= S) > sum(vec[t] * bins[t][0] * bins[t][1] for t in S):
            return 'hall'
    return None


def npos(sizes, cap):
    """own implementation of normal patterns: subset sums of `sizes` that are <= cap (bottom-left
    justified packings put each item at a sum of sizes of a chain of OTHER items in the same bin)."""
    r = {0}
    for z in sizes:
        r |= {v + z for v in r if v + z <= cap}
    return sorted(r)


def decide(bins, items, vec, tl, workers):
    m = cp_model.CpModel()
    B = [(t, k) for t in range(len(bins)) for k in range(vec[t])]
    a = {}
    for b, (t, k) in enumerate(B):
        W, H, _ = bins[t]
        fit = [i for i, (w, h) in enumerate(items) if w <= W and h <= H]
        xs, ys = [], []
        for i in fit:
            w, h = items[i]
            a[i, b] = m.NewBoolVar('')
            oth = [j for j in fit if j != i]
            x = m.NewIntVarFromDomain(cp_model.Domain.FromValues(npos([items[j][0] for j in oth], W - w)), '')
            y = m.NewIntVarFromDomain(cp_model.Domain.FromValues(npos([items[j][1] for j in oth], H - h)), '')
            xs.append(m.NewOptionalFixedSizeIntervalVar(x, w, a[i, b], ''))
            ys.append(m.NewOptionalFixedSizeIntervalVar(y, h, a[i, b], ''))
        m.AddNoOverlap2D(xs, ys)
        if NONEMPTY:
            m.AddBoolOr([a[i, b] for i in fit])
        m.AddCumulative(xs, [items[i][1] for i in fit], H)
        m.AddCumulative(ys, [items[i][0] for i in fit], W)
        m.Add(sum(items[i][0] * items[i][1] * a[i, b] for i in fit) <= W * H)
        # items taller than H/2 pairwise cannot be stacked -> disjoint x-ranges (same for wide ones)
        tall = [i for i in fit if 2 * items[i][1] > H]
        wide = [i for i in fit if 2 * items[i][0] > W]
        if len(tall) > 1:
            m.Add(sum(items[i][0] * a[i, b] for i in tall) <= W)
        if len(wide) > 1:
            m.Add(sum(items[i][1] * a[i, b] for i in wide) <= H)
        for p in range(len(fit)):
            for q in range(p + 1, len(fit)):
                i, j = fit[p], fit[q]
                if items[i][0] + items[j][0] > W and items[i][1] + items[j][1] > H:
                    m.AddBoolOr([a[i, b].Not(), a[j, b].Not()])
    for i in range(len(items)):
        m.AddExactlyOne([a[i, b] for b in range(len(B)) if (i, b) in a])
    # symmetry (empty bins allowed): relabel copies of a line by smallest item index, empty copies last;
    # then item i in copy k => copy k-1 holds an item with index < i.
    for b in range(1, len(B)):
        if B[b][0] == B[b - 1][0]:
            for (i, bb) in list(a):
                if bb == b:
                    prev = [a[j, b - 1] for j in range(i) if (j, b - 1) in a]
                    if prev:
                        m.Add(sum(prev) >= 1).OnlyEnforceIf(a[i, b])
                    else:
                        m.Add(a[i, b] == 0)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
    st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 'FEASIBLE'
    if st == cp_model.INFEASIBLE:
        return 'INFEASIBLE'
    return 'UNKNOWN'


def main():
    nm, variant, V, tot, vtl = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    lo = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # external LB: vectors with cost < lo are skipped
    t0 = time.time()
    bins, items = read_raw(nm, variant)
    assert all(c > 0 for _, _, c in bins)
    vecs = vectors(bins, len(items), V - 1)
    print(f'# {nm} {variant} target {V} raw lines {bins} vectors(cost<{V})={len(vecs)}', flush=True)
    nf = ncp = 0
    if lo:
        print(f'# skipping vectors with cost < {lo} (result then relies on that external LB)', flush=True)
    for cost, vec in vecs:
        if cost < lo:
            continue
        if filt(bins, items, vec):
            nf += 1
            continue
        rem = tot - (time.time() - t0)
        if rem < 1:
            print(f'RESULT {nm} {variant} TIMEOUT lb_proven={cost} target={V} filt={nf} cpinf={ncp}', flush=True)
            return
        t1 = time.time()
        st = decide(bins, items, vec, min(vtl, rem), workers)
        print(f'  {vec} cost {cost}: {st} {time.time() - t1:.1f}s', flush=True)
        if st == 'INFEASIBLE':
            ncp += 1
            continue
        if st == 'FEASIBLE':
            print(f'RESULT {nm} {variant} REFUTED feasible_cost={cost} < target={V}', flush=True)
            return
        print(f'RESULT {nm} {variant} UNKNOWN lb_proven={cost} target={V} filt={nf} cpinf={ncp} t={time.time()-t0:.1f}', flush=True)
        return
    print(f'RESULT {nm} {variant} CONFIRMED lb_proven={V} target={V} filt={nf} cpinf={ncp} t={time.time()-t0:.1f}', flush=True)


if __name__ == '__main__':
    main()
