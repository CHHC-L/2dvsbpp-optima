"""cg_v2: faster/stronger Dantzig-Wolfe column-generation LOWER BOUND for the 2DVSBPP
(oriented, non-guillotine, unlimited bins of <= 5 types).  Sprint 2026-10-14, track 'cg'.

Master LP (GLOP, incremental):  min sum_p c_p lam_p  s.t.  sum_{p ni i} lam_p >= 1 (all i), lam >= 0.
Columns p = (bin type t, item set S) with S 2D-packable in t (every column carries a placement).

BOUND FORMULA (Farley).  Let pi >= 0 be ANY nonnegative vector (we use the clipped GLOP duals) and
let U_t >= max { pi(S) : S 2D-packable in type t } be a PROVEN upper bound on the pricing value of
type t.  Put theta = min(1, min_t c_t / U_t).  Then pi' = theta*pi is dual feasible for the FULL
master LP (pi' >= 0 and pi'(S) <= theta*U_t <= c_t for every feasible column), so by weak duality
    LB = theta * sum_i pi_i  <=  z_LP(full)  <=  OPT.
Validity therefore depends ONLY on U_t being a true upper bound - not on GLOP accuracy, not on the
column pool, not on stabilisation.  U_t comes from (min of)
  (a) 1D relaxation: integer 0/1 knapsack DP over AREA (capacity W*H) with weights
      floor(pi_i*s) (s = integer scale) and "at most one big item" (w > W/2 and h > H/2: two such
      items can never share a bin); U = (DP + maxcard)/s, maxcard = max #items whose areas fit
      (floor error < 1/s per item);
  (b) exact CP-SAT 2D knapsack (no_overlap_2d + area + 2 cumulatives) with the side constraint
      sum w_i z_i >= c_t*s + 1 (we only look for improving columns): if INFEASIBLE, every
      feasible S has w(S) <= c_t*s;  if FEASIBLE/OPTIMAL, w(S) <= max(c_t*s, BestObjectiveBound);
      UNKNOWN (time-out, no solution) is NEVER used (falls back to (a)).  U = (that + maxcard)/s.
At CG convergence (U_t <= c_t + maxcard/s for all t) the bound equals z_LP up to ~1e-6 relative.
Final LB = smallest achievable sum of (raw-file) bin costs >= ceil(LB*(1-1e-9) - 1e-6)  (lattice
rounding; the optimum is such a sum).  An auxiliary 1D-relaxed CG (phase A, columns = area-feasible
sets with <= 1 big item, exact DP pricing) gives an independent, weaker but always-converging bound
(the 1D column set is a superset of the 2D one, so its LP value is <= the 2D LP value).

Pricing pipeline (phase B): heuristic maxrects greedy pricer (many orders, Wentges-smoothed duals)
-> per-type DP bound (skip type if it proves no improving column) -> exact CP-SAT only for the rest.
Warm start: singletons, maxrects first-fit packings of all items per type, bins of every stored
packing in results/solutions/** for this instance.  Dominated bin types (W'>=W, H'>=H, c'<=c) are
dropped for pricing (any column of a dominated type is a column of the dominating type with cost
<= c, so the master LP value is unchanged).  UB: restricted-master IP (CP-SAT) over the pool, packing
assembled from stored column placements, verified by src/check_solution.py.

usage: python src/cg_v2.py --variant area --time 60 MB_C9_11 MB_C7_11
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import vsbpp_io as io  # noqa: E402
import ledger  # noqa: E402
from cg_iplattice import ip_roundup  # noqa: E402

EPS_REL = 1e-9


# ----------------------------------------------------------------------------- instance
class Inst:
    def __init__(self, nm, area):
        self.nm = nm
        self.area = area
        self.variant = 'area' if area else 'weighted'
        self.raw_bins, self.items = io.read(nm, area)
        self.n = len(self.items)
        rb = self.raw_bins
        keep = []
        for r, (W, H, c) in enumerate(rb):
            dom = False
            for r2, (W2, H2, c2) in enumerate(rb):
                if r2 == r:
                    continue
                if W2 >= W and H2 >= H and c2 <= c and ((W2, H2, c2) != (W, H, c) or r2 < r):
                    dom = True
                    break
            if not dom:
                keep.append(r)
        # types: (W, H, c, raw index)
        self.types = [(rb[r][0], rb[r][1], rb[r][2], r) for r in keep]
        self.fit = [[i for i, (w, h) in enumerate(self.items) if w <= W and h <= H]
                    for (W, H, c, r) in self.types]
        self.all_costs = sorted(set(b[2] for b in rb))
        a = [w * h for (w, h) in self.items]
        self.areas = a
        self.maxcard = []
        for t, (W, H, c, r) in enumerate(self.types):
            s = 0; k = 0
            for v in sorted(a[i] for i in self.fit[t]):
                if s + v > W * H:
                    break
                s += v; k += 1
            self.maxcard.append(k)


def lattice_roundup(lb, costs):
    """smallest value >= ceil(lb - tol) that is a nonnegative integer combination of costs."""
    target = int(math.ceil(lb * (1 - EPS_REL) - 1e-6))
    if target <= 0:
        return 0
    N = target + max(costs) + 1
    reach = np.zeros(N, dtype=bool); reach[0] = True
    for c in sorted(set(costs)):
        # unbounded: reach[v] |= reach[v-c], processed in blocks of length c (sequential deps)
        for k in range(1, (N - 1) // c + 1):
            lo = k * c; hi = min(N, lo + c)
            reach[lo:hi] |= reach[lo - c:hi - c]
    idx = np.nonzero(reach[target:])[0]
    return int(target + idx[0])


# ----------------------------------------------------------------------------- maxrects
class MaxRects:
    __slots__ = ('free',)

    def __init__(self, W, H):
        self.free = [(0, 0, W, H)]

    def place(self, w, h):
        best = None; bx = by = 0
        for (fx, fy, fw, fh) in self.free:
            if w <= fw and h <= fh:
                k = (fy + h, fx)
                if best is None or k < best:
                    best = k; bx = fx; by = fy
        if best is None:
            return None
        self._split(bx, by, w, h)
        return (bx, by)

    def _split(self, x, y, w, h):
        x2 = x + w; y2 = y + h
        old = []; new = []
        for fr in self.free:
            fx, fy, fw, fh = fr
            fx2 = fx + fw; fy2 = fy + fh
            if x >= fx2 or x2 <= fx or y >= fy2 or y2 <= fy:
                old.append(fr); continue
            if x > fx:
                new.append((fx, fy, x - fx, fh))
            if x2 < fx2:
                new.append((x2, fy, fx2 - x2, fh))
            if y > fy:
                new.append((fx, fy, fw, y - fy))
            if y2 < fy2:
                new.append((fx, y2, fw, fy2 - y2))
        # prune: new pieces contained in any other rect; old rects contained in a new piece
        keepnew = []
        for a_i, a in enumerate(new):
            ax, ay, aw, ah = a; ax2 = ax + aw; ay2 = ay + ah
            cont = False
            for b in old:
                if ax >= b[0] and ay >= b[1] and ax2 <= b[0] + b[2] and ay2 <= b[1] + b[3]:
                    cont = True; break
            if not cont:
                for b_i, b in enumerate(new):
                    if b_i != a_i and ax >= b[0] and ay >= b[1] and ax2 <= b[0] + b[2] and ay2 <= b[1] + b[3]:
                        if b != a or b_i < a_i:
                            cont = True; break
            if not cont:
                keepnew.append(a)
        if keepnew:
            old = [b for b in old if not any(b[0] >= a[0] and b[1] >= a[1] and b[0] + b[2] <= a[0] + a[2]
                                              and b[1] + b[3] <= a[1] + a[3] for a in keepnew)]
        self.free = old + keepnew


def pack_order(W, H, items, order):
    """place items in this order greedily; returns list of (i, x, y) of placed items."""
    mr = MaxRects(W, H); out = []
    for i in order:
        w, h = items[i]
        p = mr.place(w, h)
        if p is not None:
            out.append((i, p[0], p[1]))
    return out, mr


def verify_bin(W, H, items, plc):
    for (i, x, y) in plc:
        w, h = items[i]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False
    for a in range(len(plc)):
        ia, xa, ya = plc[a]; wa, ha = items[ia]
        for b in range(a + 1, len(plc)):
            ib, xb, yb = plc[b]; wb, hb = items[ib]
            if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                return False
    return True


# ----------------------------------------------------------------------------- 1D DP bound
def dp_bound(inst, t, wts, want_set=False):
    """max sum wts over subsets of fit[t] with total area <= W*H and <= 1 big item.
    wts: integer array (>= 0).  returns (value, set or None)."""
    W, H, c, r = inst.types[t]
    cap = W * H
    small = []; big = []
    for i in inst.fit[t]:
        if wts[i] <= 0:
            continue
        w, h = inst.items[i]
        (big if (2 * w > W and 2 * h > H) else small).append(i)
    dp = np.zeros(cap + 1, dtype=np.int64)
    takes = []
    for i in small:
        a = inst.areas[i]
        cand = dp[:cap + 1 - a] + int(wts[i])
        if want_set:
            tk = np.zeros(cap + 1, dtype=bool)
            tk[a:] = cand > dp[a:]
            takes.append(tk)
        dp[a:] = np.maximum(dp[a:], cand)
    best = int(dp[cap]); bsel = None
    for j in big:
        a = inst.areas[j]
        v = int(wts[j]) + int(dp[cap - a])
        if v > best:
            best = v; bsel = j
    if not want_set:
        return best, None
    S = []
    rem = cap
    if bsel is not None:
        S.append(bsel); rem -= inst.areas[bsel]
    # reconstruct: dp[rem] is monotone in capacity, walk back items
    for k in range(len(small) - 1, -1, -1):
        if takes[k][rem]:
            S.append(small[k]); rem -= inst.areas[small[k]]
    return best, frozenset(S)


def scale_for(pi):
    mx = float(np.max(pi)) if len(pi) else 1.0
    return max(1, int(1e8 / max(mx, 1e-9)))


# ----------------------------------------------------------------------------- master LP
class Master:
    def __init__(self, n):
        self.lp = pywraplp.Solver.CreateSolver('GLOP')
        self.rows = [self.lp.Constraint(1, self.lp.infinity()) for _ in range(n)]
        self.obj = self.lp.Objective(); self.obj.SetMinimization()
        self.vars = []

    def add(self, S, c):
        v = self.lp.NumVar(0, self.lp.infinity(), '')
        for i in S:
            self.rows[i].SetCoefficient(v, 1)
        self.obj.SetCoefficient(v, float(c))
        self.vars.append(v)

    def solve(self):
        st = self.lp.Solve()
        assert st == pywraplp.Solver.OPTIMAL, st
        pi = np.array([max(0.0, r.dual_value()) for r in self.rows])
        return self.obj.Value(), pi

    def values(self):
        return np.array([v.solution_value() for v in self.vars])


# ----------------------------------------------------------------------------- phase A (1D)
def phase_a(inst, tl):
    """CG with area-relaxed columns (exact DP pricing). Returns Farley LB (valid), z, iters."""
    t0 = time.time(); n = inst.n
    M = Master(n)
    for i in range(n):
        c = min(inst.types[t][2] for t in range(len(inst.types)) if i in inst.fit[t])
        M.add([i], c)
    best = 0.0; it = 0; z = None; conv = False; best_pi = None
    while time.time() - t0 < tl:
        it += 1
        z, pi = M.solve()
        s = scale_for(pi); wts = np.floor(pi * s).astype(np.int64)
        theta = 1.0; added = 0
        for t, (W, H, c, r) in enumerate(inst.types):
            v, S = dp_bound(inst, t, wts, want_set=True)
            U = (v + inst.maxcard[t]) / s
            if U > c:
                theta = min(theta, c / U)
            if S and sum(pi[i] for i in S) > c * (1 + 1e-7):
                M.add(sorted(S), c); added += 1
        if theta * float(np.sum(pi)) > best:
            best = theta * float(np.sum(pi)); best_pi = pi.copy()
        if added == 0:
            conv = True
            break
    return best, z, it, conv, best_pi


# ----------------------------------------------------------------------------- CP pricing
class _CB(cp_model.CpSolverSolutionCallback):
    def __init__(self, z, xv, yv, cand, cap=30):
        super().__init__()
        self.z, self.xv, self.yv, self.cand, self.cap = z, xv, yv, cand, cap
        self.sols = []

    def on_solution_callback(self):
        plc = [(i, self.Value(self.xv[i]), self.Value(self.yv[i])) for i in self.cand if self.Value(self.z[i])]
        self.sols.append(plc)
        if len(self.sols) > self.cap:
            self.sols.pop(0)


def cp_price(inst, t, pi, s, wts, tl, workers, hint=None, thr_w=None):
    """returns (list of placements of improving columns, proven integer UB on w(S) or None).
    Side constraint w(S) >= base+1 with base = c_t*s (default) or thr_w; UB = max(base, CP bound)."""
    W, H, c, r = inst.types[t]
    base = c * s if thr_w is None else int(thr_w)
    cand = [i for i in inst.fit[t] if wts[i] > 0]
    if not cand:
        return [], 0
    m = cp_model.CpModel()
    z = {}; xv = {}; yv = {}; xi = []; yi = []
    for i in cand:
        w, h = inst.items[i]
        z[i] = m.NewBoolVar('')
        xv[i] = m.NewIntVar(0, W - w, ''); yv[i] = m.NewIntVar(0, H - h, '')
        xi.append(m.NewOptionalFixedSizeIntervalVar(xv[i], w, z[i], ''))
        yi.append(m.NewOptionalFixedSizeIntervalVar(yv[i], h, z[i], ''))
    m.AddNoOverlap2D(xi, yi)
    m.Add(sum(inst.areas[i] * z[i] for i in cand) <= W * H)
    m.AddCumulative(xi, [inst.items[i][1] for i in cand], H)
    m.AddCumulative(yi, [inst.items[i][0] for i in cand], W)
    obj = sum(int(wts[i]) * z[i] for i in cand)
    m.Add(obj >= base + 1)
    m.Maximize(obj)
    if hint:
        hs = {i: (x, y) for (i, x, y) in hint}
        for i in cand:
            if i in hs:
                m.AddHint(z[i], 1); m.AddHint(xv[i], hs[i][0]); m.AddHint(yv[i], hs[i][1])
            else:
                m.AddHint(z[i], 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tl
    sv.parameters.num_workers = workers
    cb = _CB(z, xv, yv, cand)
    st = sv.Solve(m, cb)
    if st == cp_model.INFEASIBLE:
        ub = base
    elif st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bb = sv.BestObjectiveBound()
        ub = max(base, int(math.floor(bb + 1e-6))) if math.isfinite(bb) else None
    else:
        ub = None
    return cb.sols, ub


# ----------------------------------------------------------------------------- phase B
def load_stored_columns(inst):
    cols = []
    files = set(glob.glob(os.path.join(ROOT, 'results', 'solutions', '**', inst.nm + '_*.json'), recursive=True))
    files |= set(glob.glob(os.path.join(ROOT, 'results', 'solutions', inst.nm + '_*.json')))
    rb = inst.raw_bins
    for f in sorted(files):
        try:
            d = json.load(open(f))
            place = d['place']
        except Exception:
            continue
        # map (w,h) -> available item indices
        pool = {}
        for i, wh in enumerate(inst.items):
            pool.setdefault(tuple(wh), []).append(i)
        bins = {}
        ok = True
        for p in place:
            tr, k, w, h, x, y = p
            lst = pool.get((w, h))
            if not lst or tr >= len(rb):
                ok = False; break
            i = lst.pop()
            bins.setdefault((tr, k), []).append((i, x, y))
        if not ok:
            continue
        for (tr, k), plc in bins.items():
            W, H = rb[tr][0], rb[tr][1]
            if not verify_bin(W, H, inst.items, plc):
                continue
            # cheapest kept type that contains this bin geometrically
            best = None
            for t, (W2, H2, c2, r2) in enumerate(inst.types):
                if W2 >= W and H2 >= H and (best is None or c2 < inst.types[best][2]):
                    best = t
            if best is not None:
                cols.append((best, plc))
    return cols


def run(inst, tl=60.0, tl_price=5.0, workers=1, seed=0, alpha=0.5, rmp_time=15.0, log=print, pi0=None, lb0=-1.0,
        stop_ub=None):
    """pi0/lb0: optional stability centre (e.g. phase-A duals; any pi works, it only steers pricing)."""
    t0 = time.time(); n = inst.n; T = len(inst.types)
    rng = random.Random(seed)
    items = inst.items
    pool = []          # (t, frozenset S, placement)
    seen = set()
    M = Master(n)

    def add_col(t, plc):
        S = frozenset(i for (i, x, y) in plc)
        if not S or (t, S) in seen:
            return False
        seen.add((t, S)); pool.append((t, S, tuple(plc))); M.add(sorted(S), inst.types[t][2])
        return True

    # warm start
    for i in range(n):
        tb = min((inst.types[t][2], t) for t in range(T) if i in inst.fit[t])[1]
        add_col(tb, [(i, 0, 0)])
    nst = 0
    for t, plc in load_stored_columns(inst):
        nst += add_col(t, plc)
    for t, (W, H, c, r) in enumerate(inst.types):     # first-fit maxrects of all fitting items
        for key in (lambda i: (-items[i][1], -items[i][0]), lambda i: -inst.areas[i]):
            rest = sorted(inst.fit[t], key=key)
            while rest:
                plc, _ = pack_order(W, H, items, rest)
                add_col(t, plc)
                got = set(i for i, _, _ in plc)
                rest = [i for i in rest if i not in got]
    log(f'  warm start: {len(pool)} cols ({nst} from stored packings)')

    best_lb = 0.0; pi_center = pi0; best_center_lb = lb0
    it = 0; n_cp = 0; t_cp = 0.0; conv = False; z = None; closed_early = False
    last_ipchk = 0.0; pi = None
    if stop_ub is not None and lb0 > 0 and max(lattice_roundup(lb0, inst.all_costs),
                                               ip_roundup(inst.nm, inst.area, lb0, pis=[pi0]) or 0) >= stop_ub:
        tl = 0.0; closed_early = True      # phase-A bound already closes the instance
    tlp = dict((t, tl_price) for t in range(T))
    proved_at = {}     # type -> pool size at which 'no improving column' was proven (same LP => same pi)
    while time.time() - t0 < tl:
        it += 1
        z, pi = M.solve()
        # --- Farley bound with DP upper bounds at pi (cheap, every iteration)
        s = scale_for(pi); wts = np.floor(pi * s).astype(np.int64)
        U = []
        for t in range(T):
            v, _ = dp_bound(inst, t, wts)
            U.append((v + inst.maxcard[t]) / s)
        theta = min([1.0] + [inst.types[t][2] / U[t] for t in range(T) if U[t] > 0])
        lbF = theta * float(np.sum(pi))
        if lbF > best_lb:
            best_lb = lbF
            if stop_ub is not None and time.time() - last_ipchk > 3.0:
                last_ipchk = time.time()
                if max(lattice_roundup(max(best_lb, lb0), inst.all_costs),
                       ip_roundup(inst.nm, inst.area, max(best_lb, lb0), pis=[pi0, pi]) or 0) >= stop_ub:
                    closed_early = True
                    break
        if lbF > best_center_lb:
            best_center_lb = lbF; pi_center = pi.copy()
        # --- heuristic pricing (smoothed duals first, then plain)
        added = 0
        for a in ((alpha, 0.0) if pi_center is not None and alpha > 0 else (0.0,)):
            ps = a * pi_center + (1 - a) * pi if a > 0 else pi
            added = heur_price(inst, ps, pi, add_col, rng)
            if added:
                break
        if added:
            continue
        # --- exact phase: DP filter then CP-SAT
        Uex = list(U); found = 0; allproved = True
        order = sorted(range(T), key=lambda t: -(U[t] - inst.types[t][2]))
        for t in order:
            W, H, c, r = inst.types[t]
            if U[t] <= c * (1 + EPS_REL) + inst.maxcard[t] / s:
                continue     # DP proves no improving column (up to rounding slack)
            if proved_at.get(t, (-1,))[0] == len(pool):
                Uex[t] = min(Uex[t], proved_at[t][1]); continue
            if time.time() - t0 > tl:
                allproved = False; break
            ta = time.time()
            sols, ubw = cp_price(inst, t, pi, s, wts, min(tlp[t], max(0.5, tl - (time.time() - t0))), workers)
            t_cp += time.time() - ta; n_cp += 1
            if ubw is not None:
                Uex[t] = min(Uex[t], (ubw + inst.maxcard[t]) / s)
                if ubw <= c * s:
                    proved_at[t] = (len(pool), Uex[t])
            if ubw is None or ubw > c * s:
                allproved = allproved and False
            for plc in sols:
                if sum(pi[i] for (i, x, y) in plc) > c * (1 + 1e-9):
                    found += add_col(t, plc)
            if ubw is None and not sols:
                tlp[t] = min(4 * tlp[t], 120.0)
            if found:
                break   # re-optimise the master before pricing other types
        if not found:
            theta = min([1.0] + [inst.types[t][2] / Uex[t] for t in range(T) if Uex[t] > 0])
            lbx = theta * float(np.sum(pi))
            best_lb = max(best_lb, lbx)
            if allproved:
                conv = True
                break
            if all(tlp[t] >= 120.0 for t in range(T)):
                break
    lb_round = lattice_roundup(best_lb, inst.all_costs)
    res = dict(name=inst.nm, variant=inst.variant, n=n, z_lp=z, lb=best_lb, lb_round=lb_round,
               iters=it, cols=len(pool), n_cp=n_cp, t_cp=round(t_cp, 1), converged=conv, closed_early=closed_early,
               t_cg=round(time.time() - t0, 1))
    res['piB'] = None if pi is None else [float(v) for v in pi]
    return res, pool, M


def heur_price(inst, ps, pi, add_col, rng, n_rand=3, thr=None):
    """greedy maxrects pricer at duals ps; adds columns improving w.r.t. LP duals pi
    (improving = pi(S) > c_t, or > thr[t] when a per-type threshold list is given; thr[t] None = skip)."""
    items = inst.items; added = 0
    for t, (W, H, c, r) in enumerate(inst.types):
        if thr is not None:
            if thr[t] is None:
                continue
            c = thr[t]
        cand = [i for i in inst.fit[t] if ps[i] > 1e-12]
        if not cand:
            continue
        dens = {i: ps[i] / inst.areas[i] for i in cand}
        orders = [sorted(cand, key=lambda i: -dens[i]), sorted(cand, key=lambda i: -ps[i])]
        for _ in range(n_rand):
            orders.append(sorted(cand, key=lambda i: -dens[i] * rng.uniform(0.6, 1.4)))
        tried = set()
        for o in orders:
            # select by order up to area capacity, pack big-first, then fill
            for fill in (1.0, 0.85):
                sel = []; ar = 0
                for i in o:
                    if ar + inst.areas[i] <= fill * W * H:
                        sel.append(i); ar += inst.areas[i]
                key = frozenset(sel)
                if key in tried:
                    continue
                tried.add(key)
                for pk in (lambda i: (-items[i][1], -items[i][0]), lambda i: -inst.areas[i]):
                    seq = sorted(sel, key=pk)
                    plc, mr = pack_order(W, H, items, seq)
                    got = set(i for i, _, _ in plc)
                    for i in o:                      # fill leftover space in order
                        if i not in got:
                            p = mr.place(*items[i])
                            if p is not None:
                                plc.append((i, p[0], p[1])); got.add(i)
                    val = sum(pi[i] for i in got)
                    if val > c * (1 + 1e-9):
                        if add_col(t, plc):
                            added += 1
    return added


# ----------------------------------------------------------------------------- UB heuristics
def assemble(inst, chosen):
    """chosen: list of (t, S, plc) covering all items -> (cost, place rows with RAW type index)."""
    done = set(); place = []; kcount = {}
    for (t, S, plc) in chosen:
        W, H, c, r = inst.types[t]
        k = kcount.get(r, 0); kcount[r] = k + 1
        for (i, x, y) in plc:
            if i in done:
                continue
            done.add(i)
            w, h = inst.items[i]
            place.append([r, k, w, h, x, y])
    assert len(done) == inst.n
    return sum(inst.types[t][2] for (t, S, plc) in chosen), place


def dive_ub(inst, pool, tl, seed=1, max_heur=2):
    """price-and-dive: repeatedly fix the column with the largest LP value (covered rows -> rhs 0),
    re-optimise with heuristic pricing.  Pure heuristic (UB only).  Returns (cost, chosen, cols)."""
    t0 = time.time(); n = inst.n; rng = random.Random(seed)
    M = Master(n); cols = list(pool); seen = set((t, S) for (t, S, _) in cols)
    for (t, S, plc) in cols:
        M.add(sorted(S), inst.types[t][2])

    def add_col(t, plc):
        S = frozenset(i for (i, x, y) in plc)
        if not S or (t, S) in seen:
            return False
        seen.add((t, S)); cols.append((t, S, tuple(plc))); M.add(sorted(S), inst.types[t][2])
        return True
    covered = set(); fixed = []; fixed_set = set()
    while len(covered) < n:
        for _ in range(max_heur if time.time() - t0 < tl else 0):
            z, pi = M.solve()
            if not heur_price(inst, pi, pi, add_col, rng, n_rand=1):
                break
        M.solve()
        lam = M.values()
        best = None
        for p, (t, S, plc) in enumerate(cols):
            if p in fixed_set:
                continue
            newi = len(S - covered)
            if newi == 0:
                continue
            key = (round(lam[p], 9), newi)
            if best is None or key > best[0]:
                best = (key, p)
        p = best[1]; fixed.append(p); fixed_set.add(p)
        M.vars[p].SetLb(1.0)
        for i in cols[p][1]:
            if i not in covered:
                covered.add(i); M.rows[i].SetLb(0.0)
    chosen = [cols[p] for p in fixed]
    return sum(inst.types[t][2] for (t, S, plc) in chosen), chosen, cols


def rmp_ub(inst, pool, lb, ub_cap, tl, workers):
    m = cp_model.CpModel(); lam = [m.NewBoolVar('') for _ in pool]
    cover = [[] for _ in range(inst.n)]
    for p, (t, S, plc) in enumerate(pool):
        for i in S:
            cover[i].append(lam[p])
    for i in range(inst.n):
        m.AddBoolOr(cover[i])
    obj = sum(int(inst.types[t][2]) * lam[p] for p, (t, S, plc) in enumerate(pool))
    m.Add(obj >= int(lb))
    if ub_cap is not None:
        m.Add(obj <= int(ub_cap))
    m.Minimize(obj)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = workers
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None
    chosen = [p for p in range(len(pool)) if sv.Value(lam[p])]
    done = set(); place = []; kcount = {}
    for p in chosen:
        t, S, plc = pool[p]
        W, H, c, r = inst.types[t]
        k = kcount.get(r, 0); kcount[r] = k + 1
        for (i, x, y) in plc:
            if i in done:
                continue
            done.add(i)
            w, h = inst.items[i]
            place.append([r, k, w, h, x, y])
    assert len(done) == inst.n
    cost = sum(inst.types[pool[p][0]][2] for p in chosen)
    return cost, place


def check(nm, solf, variant):
    out = subprocess.run([sys.executable, os.path.join(SRC, 'check_solution.py'),
                          os.path.join(ROOT, 'data', 'instances', nm + '.txt'), solf, variant],
                         capture_output=True, text=True, timeout=120)
    txt = out.stdout.strip()
    if out.returncode == 0 and txt.startswith('valid'):
        return int(txt.split('cost')[-1].strip())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variant', choices=['weighted', 'area'], default='weighted')
    ap.add_argument('--time', type=float, default=60.0, help='CG time limit per instance (s)')
    ap.add_argument('--price-time', type=float, default=5.0, help='initial CP pricing limit (s)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--rmp-time', type=float, default=15.0)
    ap.add_argument('--dive-time', type=float, default=0.0, help='price-and-dive UB heuristic time (0 = off)')
    ap.add_argument('--phase-a', type=float, default=10.0, help='time for 1D-relaxed CG (0 = skip)')
    ap.add_argument('--alpha', type=float, default=0.5, help='Wentges smoothing weight on the centre')
    ap.add_argument('--no-ledger', action='store_true')
    ap.add_argument('--skip-done', action='store_true', help='skip names already present in --out')
    ap.add_argument('--stop-at', default=None, help='local HH:MM (next occurrence); no new instance after it, '
                    'and per-instance limits are cut so that the run ends by then')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'cg', 'cg_v2_runs.jsonl'))
    ap.add_argument('names', nargs='+')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    soldir = os.path.join(ROOT, 'results', 'solutions', 'sprint')
    os.makedirs(soldir, exist_ok=True)
    stop_ts = None
    if a.stop_at:
        hh, mm = map(int, a.stop_at.split(':'))
        lt = time.localtime(); base = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        stop_ts = base if base > time.time() - 6 * 3600 else base + 86400
        if stop_ts - time.time() > 12 * 3600:
            stop_ts -= 86400
    done = set()
    if a.skip_done and os.path.exists(a.out):
        done = set(json.loads(l)['name'] for l in open(a.out) if l.strip())
    for nm in a.names:
        if nm in done:
            continue
        T0 = time.time()
        if stop_ts is not None:
            left = stop_ts - T0
            if left < 20:
                print('stop-at reached; skipping', nm, flush=True); break
            a.time = min(a.time, max(5.0, left - a.phase_a - a.rmp_time - a.dive_time - 5))
        inst = Inst(nm, a.variant == 'area')
        try:
            bst = ledger.best().get((nm, inst.variant), {})
        except Exception:
            bst = {}
        stop_ub = bst.get('ub') if bst.get('ub_checked') else None
        lbA = None; piA = None; convA = None
        if a.phase_a > 0:
            lbA, zA, itA, convA, piA = phase_a(inst, a.phase_a)
        res, pool, M = run(inst, tl=a.time, tl_price=a.price_time, workers=a.workers,
                           log=lambda s: print(s, flush=True), pi0=piA, lb0=(lbA if lbA else -1.0),
                           alpha=a.alpha, stop_ub=stop_ub)
        res['convA'] = convA
        lb = max(res['lb'], lbA or 0.0)
        res['lbA'] = lbA
        res['lb_lattice'] = lattice_roundup(lb, inst.all_costs)
        res['piA'] = None if piA is None else [float(v) for v in piA]
        lbi = ip_roundup(nm, inst.area, lb, pis=[piA, res['piB']])
        res['lb_ip'] = lbi
        res['lb_round'] = max(res['lb_lattice'], lbi or 0)
        cap = bst.get('ub')
        ub = place = None
        if a.dive_time > 0:
            try:
                dcost, dch, pool = dive_ub(inst, pool, a.dive_time)
                ub, place = assemble(inst, dch)
            except Exception as e:  # heuristic only; never fatal
                print('  dive failed', repr(e), flush=True)
        res['dive_ub'] = ub
        if a.rmp_time > 0 and (ub is None or ub > res['lb_round']):
            u2, p2 = rmp_ub(inst, pool, res['lb_round'], ub, a.rmp_time, a.workers)
            if u2 is not None and (ub is None or u2 < ub):
                ub, place = u2, p2
        res['rmp_ub'] = ub
        solrel = ''; checked = 0
        if ub is not None:
            solrel = ledger.sol_relpath(nm, inst.variant, 'cg_v2_root', ub)
            json.dump(dict(name=nm, variant=inst.variant, method='cg_v2_root', cost=ub, place=place),
                      open(os.path.join(ROOT, solrel), 'w'))
            cv = check(nm, os.path.join(ROOT, solrel), inst.variant)
            checked = int(cv == ub)
            if not checked:
                print('  !! RMP packing failed check', cv, ub, flush=True)
        res['ub_checked'] = checked
        res['ledger_ub'] = cap; res['ledger_ub_checked'] = bst.get('ub_checked')
        best_ub = min([v for v in (cap if bst.get('ub_checked') else None, ub if checked else None) if v is not None],
                      default=None)
        res['best_checked_ub'] = best_ub
        if best_ub is not None and res['lb_round'] > best_ub:
            print('  !!! LB > checked UB -- BUG, not recording LB', flush=True)
            res['BUG'] = True
        res['time'] = round(time.time() - T0, 1); res['threads'] = a.workers
        print(json.dumps(res), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
        if not a.no_ledger and not res.get('BUG'):
            closed = best_ub is not None and res['lb_round'] == best_ub
            status = 'OPTIMAL' if (checked and ub == res['lb_round']) else 'LB_ONLY' if not checked else 'FEASIBLE'
            note = (f"zlp={(res['z_lp'] or 0):.3f} lbF={res['lb']:.3f} lbA={(lbA or 0):.3f} conv={res['converged']} "
                    f"it={res['iters']} cols={res['cols']} ncp={res['n_cp']} early={res['closed_early']}" + (' closes_vs_best_checked_ub' if closed else ''))
            ledger.record(nm, inst.variant, 'cg_v2_root', status, lb=res['lb_round'],
                          ub=ub if checked else None, time_s=res['time'], threads=a.workers,
                          sol_file=solrel if checked else '', checked=checked, note=note)


if __name__ == '__main__':
    main()
