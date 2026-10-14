"""Integer 'type-count' rounding of a valid real lower bound (track cg, sprint 2026-10-14).

Given a VALID real lower bound L on OPT (e.g. the cg_v2 Farley/LP bound), any optimal packing uses
n_t >= 0 bins of each (distinct, raw-file) bin type t and satisfies
   (1) sum_t c_t n_t >= ceil(L*(1-1e-9) - 1e-6)                (it costs at least L, costs integer)
   (2) for every nonempty subset Q of types:  sum_{t in Q} W_t H_t n_t >= a(Q),
       a(Q) = total area of the items that fit (oriented) in NO type outside Q
       (those items must lie in bins of types in Q, and items in a bin cannot overlap).
   (3) for every nonempty subset Q of types:  sum_{t in Q} n_t >= b(Q),
       b(Q) = number of items i that fit in NO type outside Q and are 'big' (w_i > W_t/2 and
       h_i > H_t/2) in EVERY type t in Q in which they fit: such an item must lie in a bin of a type
       t in Q where it is big, and two items that are both big in t overlap (their x-intervals
       both contain the bin's mid-line x = W_t/2, likewise y), so each bin holds <= 1 of them.
   (4) generalisation of (3): K = a set of items that fit in NO type outside Q and are pairwise
       CONFLICTING in Q (for every t in Q in which both fit: w_i + w_j > W_t and h_i + h_j > H_t, so
       their x-intervals and y-intervals must both overlap in any bin of type t): every bin holds
       <= 1 item of K, hence sum_{t in Q} n_t >= |K|.  K is found greedily (any K is valid).
   (5) dual-weighted count cuts: for ANY vector pi >= 0 and P_t >= max{pi(S): S packable in t},
       sum_t P_t n_t >= sum_i pi_i  (each item lies in exactly one bin, and a bin of type t holds a
       packable set, so sum_i pi_i = sum_bins pi(bin) <= sum_t P_t n_t).  P_t is computed exactly in
       integers: weights w_i = floor(pi_i * s), P^w_t = DP_t(w) + maxcard_t where DP_t = 0/1 knapsack
       over AREA (cap W_t H_t, <= 1 item big in t) - a relaxation of 2D packability - and maxcard_t
       = max #items whose areas fit; the cut is  sum_t P^w_t n_t >= sum_i w_i  (valid because
       s*pi(S) < w(S) + |S| <= P^w_t and sum_i w_i <= s * sum_i pi_i).  Used with the CG duals
       (phase A / phase B) and with pi = item areas.
So  LB_int = min { sum c_t n_t : (1)-(5), n integer >= 0 }  <= OPT.
Benders strengthening (benders > 0): let n* be the IP minimiser.  Solve the separation LP
   max sum_i pi_i - sum_t n*_t mu_t  s.t. pi(S) <= mu_t for all area-feasible (t, S), 0 <= pi <= 1
by column generation (exact DP pricing).  For the resulting pi the cut (5) is recomputed EXACTLY in
integers (DP) and added only if it is violated by n*; then re-solve.  Every cut is an instance of (5),
so every IP value along the way is a valid LB (we return the last OPTIMAL one).  It is >= the plain lattice
rounding (which is the same IP without (2)).  Solved exactly with CP-SAT (<= 5 integer variables,
<= 31 constraints); the value is used ONLY if CP-SAT reports OPTIMAL, otherwise we fall back to the
plain lattice rounding.  All distinct raw bin lines are used (dominated ones too), so nothing relies
on type-dominance arguments.
"""
import math
import os
import sys
import time
from itertools import combinations

import numpy as np

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402


def dp_price(W, H, items, wts):
    """integer upper bound on max w(S) over S packable in W x H: area knapsack, <= 1 big item."""
    cap = W * H
    dp = np.zeros(cap + 1, dtype=np.int64); big = []
    for (w, h), v in zip(items, wts):
        if v <= 0 or w > W or h > H:
            continue
        if 2 * w > W and 2 * h > H:
            big.append((w * h, int(v))); continue
        a = w * h
        dp[a:] = np.maximum(dp[a:], dp[:cap + 1 - a] + int(v))
    best = int(dp[cap])
    for a, v in big:
        best = max(best, v + int(dp[cap - a]))
    return best


def dp_argmax(W, H, items, pi, allowed):
    """float version with reconstruction: max pi(S), S area-feasible in W x H with <= 1 big item."""
    cap = W * H; small = []; big = []
    for i in allowed:
        w, h = items[i]
        if pi[i] <= 1e-12:
            continue
        (big if (2 * w > W and 2 * h > H) else small).append(i)
    dp = np.zeros(cap + 1); takes = []
    for i in small:
        a = items[i][0] * items[i][1]
        cand = dp[:cap + 1 - a] + pi[i]
        tk = np.zeros(cap + 1, dtype=bool); tk[a:] = cand > dp[a:] + 1e-15
        takes.append(tk)
        dp[a:] = np.maximum(dp[a:], cand)
    best = dp[cap]; bsel = None
    for j in big:
        a = items[j][0] * items[j][1]
        if pi[j] + dp[cap - a] > best:
            best = pi[j] + dp[cap - a]; bsel = j
    S = []; rem = cap
    if bsel is not None:
        S.append(bsel); rem -= items[bsel][0] * items[bsel][1]
    for k in range(len(small) - 1, -1, -1):
        if takes[k][rem]:
            S.append(small[k]); rem -= items[small[k]][0] * items[small[k]][1]
    return float(sum(pi[i] for i in S)), S


def exact_cut(types, items, fits, pi):
    """integer cut (5) for vector pi: (coef, rhs, wts) with sum_t coef_t n_t >= rhs valid, where
    wts are the integer item weights the cut was computed from (rhs = sum(wts))."""
    pi = np.maximum(np.asarray(pi, dtype=float), 0.0)
    if pi.max() <= 0:
        return None
    sc = max(1, int(1e8 / pi.max()))
    wts = np.floor(pi * sc).astype(np.int64)
    coef = []
    for t, (W, H, c) in enumerate(types):
        ar = sorted(w * h for (w, h), f in zip(items, fits) if f[t])
        k = 0; acc = 0
        for v in ar:
            if acc + v > W * H:
                break
            acc += v; k += 1
        coef.append(dp_price(W, H, items, wts) + k)
    return coef, int(wts.sum()), [int(v) for v in wts]


def separate(types, items, fits, nstar, tl):
    """CG on the separation LP; returns a cut violated by nstar or None."""
    from ortools.linear_solver import pywraplp
    t0 = time.time(); n = len(items); T = len(types)
    lp = pywraplp.Solver.CreateSolver('GLOP')
    cov = [lp.Constraint(1, lp.infinity()) for _ in range(n)]
    cnt = [lp.Constraint(-lp.infinity(), float(nstar[t])) for t in range(T)]
    obj = lp.Objective(); obj.SetMinimization()
    for i in range(n):
        u = lp.NumVar(0, lp.infinity(), ''); cov[i].SetCoefficient(u, 1); obj.SetCoefficient(u, 1.0)
    seen = set()
    while time.time() - t0 < tl:
        if lp.Solve() != pywraplp.Solver.OPTIMAL:
            return None
        pi = np.array([max(0.0, r.dual_value()) for r in cov])
        mu = [max(0.0, -r.dual_value()) for r in cnt]
        cut = exact_cut(types, items, fits, pi)
        if cut is not None and sum(cut[0][t] * nstar[t] for t in range(T)) < cut[1]:
            return cut
        added = 0
        for t, (W, H, c) in enumerate(types):
            if nstar[t] == 0:
                continue
            val, S = dp_argmax(W, H, items, pi, [i for i in range(n) if fits[i][t]])
            if val > mu[t] + 1e-9 and (t, tuple(sorted(S))) not in seen:
                seen.add((t, tuple(sorted(S))))
                v = lp.NumVar(0, lp.infinity(), '')
                for i in S:
                    cov[i].SetCoefficient(v, 1)
                cnt[t].SetCoefficient(v, 1)
                added += 1
        if not added:
            return None
    return None


def shrink(coef, rhs, cap=10 ** 6):
    """valid weakening of sum coef_t n_t >= rhs (n >= 0) to small integers: ceil(coef/g), floor(rhs/g)."""
    g = max(1, max(max(coef), 1) // cap)
    return [-(-int(a) // g) for a in coef], int(rhs) // g


def ip_roundup(nm, area, lb_real, ub_cap=None, tl=10.0, pis=(), benders=0.0, info=None, separator=None, log=None,
               checker=None):
    """returns the IP value (a valid LB) or None. If info is a dict, info['cert'] receives everything
    needed to rebuild and re-solve the final IP offline: the types, the target of (1), every cut (5)
    with the integer item weights it came from, and every fixed-count no-good (n <= n* infeasible)."""
    bins, items = io.read(nm, area)
    types = sorted(set(bins))
    T = len(types)
    target = int(math.ceil(lb_real * (1 - 1e-9) - 1e-6))
    fits = [[w <= W and h <= H for (W, H, c) in types] for (w, h) in items]
    cuts = []
    vecs = [np.array([w * h for (w, h) in items], dtype=float)] + [np.asarray(p, dtype=float) for p in pis if p is not None]
    for pi in vecs:
        pi = np.maximum(pi, 0.0)
        if pi.max() <= 0:
            continue
        sc = max(1, int(1e8 / pi.max()))
        wts = np.floor(pi * sc).astype(np.int64)
        coef = []
        for t, (W, H, c) in enumerate(types):
            ar = sorted(w * h for (w, h), f in zip(items, fits) if f[t])
            k = 0; acc = 0
            for v in ar:
                if acc + v > W * H:
                    break
                acc += v; k += 1
            coef.append(dp_price(W, H, items, wts) + k)
        cuts.append((coef, int(wts.sum()), [int(v) for v in wts]))
    cert = dict(name=nm, variant='area' if area else 'weighted', lb_real=float(lb_real), target=target,
                types=[list(t) for t in types], cuts=[], nogoods=[], val=None)
    if info is not None:
        info['cert'] = cert
    m = cp_model.CpModel()
    # domain n_t <= #items fitting in t: an optimal packing has no empty bin (costs > 0), so its
    # count vector lies in this box and the box minimum is <= OPT.
    nv = [m.NewIntVar(0, sum(1 for f in fits if f[t]), '') for t in range(T)]
    cost = sum(c * nv[t] for t, (W, H, c) in enumerate(types))
    m.Add(cost >= target)
    for r in range(1, T + 1):
        for Q in combinations(range(T), r):
            Qs = set(Q)
            aQ = sum(w * h for (w, h), f in zip(items, fits) if not any(f[t] for t in range(T) if t not in Qs))
            if aQ > 0:
                m.Add(sum(types[t][0] * types[t][1] * nv[t] for t in Q) >= aQ)
            bQ = 0
            for (w, h), f in zip(items, fits):
                if any(f[t] for t in range(T) if t not in Qs):
                    continue
                if all((2 * w > types[t][0] and 2 * h > types[t][1]) for t in Q if f[t]):
                    bQ += 1
            if bQ > 0:
                m.Add(sum(nv[t] for t in Q) >= bQ)
            cand = [i for i, f in enumerate(fits) if not any(f[t] for t in range(T) if t not in Qs)]

            def conf(i, j):
                (wi, hi), (wj, hj) = items[i], items[j]
                return all(wi + wj > types[t][0] and hi + hj > types[t][1]
                           for t in Q if fits[i][t] and fits[j][t])
            kbest = 0
            for key in (lambda i: -items[i][0] * items[i][1], lambda i: -items[i][0], lambda i: -items[i][1]):
                K = []
                for i in sorted(cand, key=key):
                    if all(conf(i, j) for j in K):
                        K.append(i)
                kbest = max(kbest, len(K))
            if kbest > bQ:
                m.Add(sum(nv[t] for t in Q) >= kbest)
    for coef, rhs, wts in cuts:
        cert['cuts'].append(dict(src='initial', coef=[int(a) for a in coef], rhs=rhs, wts=wts))
        coef, rhs = shrink(coef, rhs)
        m.Add(sum(coef[t] * nv[t] for t in range(T)) >= rhs)
    m.Minimize(cost)
    t0 = time.time(); val = None; nb = 0
    while True:
        s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 1
        st = s.Solve(m)
        if st != cp_model.OPTIMAL:
            break      # keep the last OPTIMAL value (each is a valid LB)
        val = int(round(s.ObjectiveValue()))
        cert['val'] = val
        if benders <= 0 or time.time() - t0 > benders or (ub_cap is not None and val >= ub_cap):
            break
        nstar = [int(s.Value(v)) for v in nv]
        if log:
            log(f'    IP value {val} at n*={nstar} ({time.time() - t0:.1f}s, {nb} cuts)')
        cut = (separator or separate)(types, items, fits, nstar, max(1.0, benders - (time.time() - t0)))
        if cut is None:
            # n* passes the LP test.  Optional exact check: can the items be packed into <= n*_t bins of each
            # type?  INFEASIBLE => every n <= n* (componentwise) is infeasible => no-good OR_t (n_t >= n*_t + 1).
            if checker is None or time.time() - t0 > benders:
                break
            verdict, payload = checker(types, nstar, max(1.0, benders - (time.time() - t0)))
            if log:
                log(f'    exact check of n*={nstar}: {verdict}')
            if verdict == 'feasible':
                if info is not None:
                    info['packing'] = payload; info['packing_cost'] = val
                break
            if verdict != 'infeasible':
                break
            bs = [m.NewBoolVar('') for _ in range(T)]
            for t in range(T):
                m.Add(nv[t] >= nstar[t] + 1).OnlyEnforceIf(bs[t])
            m.AddBoolOr(bs)
            cert['nogoods'].append([int(v) for v in nstar])
            nb += 1
            if info is not None:
                info['nogoods'] = info.get('nogoods', 0) + 1
            continue
        nb += 1
        cert['cuts'].append(dict(src='benders', coef=[int(a) for a in cut[0]], rhs=int(cut[1]),
                                 wts=list(cut[2]) if len(cut) > 2 else None))
        cc, rr = shrink(cut[0], cut[1])
        m.Add(sum(cc[t] * nv[t] for t in range(T)) >= rr)
    if info is not None:
        info['benders_cuts'] = nb
    return val   # None: INFEASIBLE (cannot happen for a valid L) or time-out -> caller falls back


if __name__ == '__main__':
    v = sys.argv[1]; nm = sys.argv[2]; L = float(sys.argv[3]); B = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    inf = {}
    print(ip_roundup(nm, v == 'area', L, benders=B, info=inf), inf)
