"""verify_cg_safe: numerically safe re-derivation of the column-generation lower bounds (2026-10-14).

Independent of cg_v2 / cg_iplattice / cg_bendex: imports only vsbpp_io and ledger; every helper below
is written from scratch.  Those programs are used only as *sources of numbers* (dual vectors, Benders
cut weights, no-good vectors), read from their output files; nothing they computed is trusted.

Claim (name, variant, LB).  An optimal packing uses n_t >= 0 bins of every distinct raw bin line t
(no empty bin, so n_t <= #items fitting t).  For integer item weights w >= 0 and integers
P_t >= max{ w(S) : S subset of items, S packable in W_t x H_t, no rotation }, every packing satisfies
        (C)   sum_t P_t n_t >= sum_i w_i
because each item lies in exactly one bin and a bin of type t holds a packable set.  Weights come from
(a) item areas, (b) LP dual vectors pi >= 0 of cg_v2 / cg_bendex (w_i = floor(pi_i * s)), (c) the
integer weights 'wts' logged with every cg_bendex cut.  P_t = min of
  * DP_t: exact 0/1 knapsack over AREA (capacity W_t H_t) with at most one item that is 'big' in t
    (w > W_t/2 and h > H_t/2: two of them overlap at the bin centre) - a relaxation of packability;
  * the proven bound of an independent CP-SAT 2D knapsack (NoOverlap2D, integer objective):
    OPTIMAL value, or ceil(BestObjectiveBound) on time-out; with a target K (the coefficient a
    cg_bendex cut claims) the model 'w(S) >= K+1' is solved instead: INFEASIBLE => P_t = K.
Combinatorial constraints, re-derived here (Q = any nonempty set of types, I(Q) = items that fit in
no type outside Q, so they lie in bins of types in Q):
  (A) area(I(Q)) <= sum_{t in Q} W_t H_t n_t ;
  (K) for a set K subset of I(Q) of items pairwise conflicting in Q (for every t in Q where both fit:
      w_i + w_j > W_t and h_i + h_j > H_t, so their x- and y-intervals both overlap): every bin of a
      type in Q holds <= 1 item of K, so |K| <= sum_{t in Q} n_t.
No-goods (cg_bendex 'n <= n* is infeasible'): each is re-proven INFEASIBLE by an independent
fixed-slot CP-SAT model (n*_t slots of type t, NoOverlap2D per slot, every item in exactly one slot;
symmetry breaking: items ranked by (area desc, index); slot k of a type only takes items of rank >= k,
valid because the used bins of a type can be ordered by their smallest rank).  Only re-proven ones
enter the IP, as OR_t (n_t >= n*_t + 1).
safe LB = min { sum_t c_t n_t : n integer in the box, (C) for every weight vector, (A), (K), no-goods },
solved by CP-SAT (integers only) and, when small enough, cross-checked by an exhaustive pure-Python
enumeration of all count vectors cheaper than the CP-SAT value.

Inputs come from re-runs made for this check (results/verify_cg_safe/rerun_driver.sh):
  cg_v2.py --no-ledger --rmp-time 0 (-> rerun.jsonl, piA/piB) and cg_bendex.py --no-ledger --time 90
  (-> bendex.jsonl and results/cg/certs/*.json: cut weights, no-goods, piA/piB), plus the stored piA/piB in
  results/cg/*.jsonl.  Every result row also runs a consistency test on the table's checked packing.

usage:
  cg_safe.py claims                      -> results/verify_cg_safe/claims.csv
  cg_safe.py names KIND VARIANT [CLOSED]  -> print claim names (KIND root|bendex) for a rerun batch
  cg_safe.py verify [--workers 2] [--tl 15] [--budget 600] [--tag T] [--target-rerun] [--no-wait] [names...]
                                         -> appends to results/verify_cg_safe/verify.jsonl
  cg_safe.py summary [--record]          -> summary.csv; --record writes SAFE rows to the ledger
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys
import time
from itertools import combinations

import numpy as np
from ortools.sat.python import cp_model

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import vsbpp_io as io  # noqa: E402
import ledger  # noqa: E402

ROOT = io.ROOT
OUT = os.path.join(ROOT, 'results', 'verify_cg_safe')
CLAIMS = os.path.join(OUT, 'claims.csv')
VERIFY = os.path.join(OUT, 'verify.jsonl')
RELIES = re.compile(r'relies on ledger LB (\d+) from (\S+)')


# ----------------------------------------------------------------------------- claims
def build_claims():
    """every (name, variant) of results/final/results.csv whose lb_families contains 'cg', with the cg LB
    the table relies on: the final LB if a cg_v2_* row reaches it, else the LB X that the lattice rows at
    the final LB rely on ('relies on ledger LB X from M', followed back to a cg_v2_* method)."""
    rows = ledger.rows()
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['name'], r['variant'])].append(r)
    out = []
    for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'final', 'results.csv'))):
        if 'cg' not in r['lb_families'].split():
            continue
        k = (r['name'], r['variant'])
        lb = int(r['lb'])
        L = by[k]

        def cg_rows(need):
            return [x for x in L if x['method'] in ('cg_v2_root', 'cg_v2_bendex') and x['lb'] and int(x['lb']) >= need]

        def resolve(need, meth, depth=0):
            """smallest cg LB that a source of method meth with LB >= need relies on"""
            if meth in ('cg_v2_root', 'cg_v2_bendex'):
                return need, meth
            if depth > 5:
                return None
            best = None
            for x in L:
                if x['method'] == meth and x['lb'] and int(x['lb']) >= need:
                    mm = RELIES.search(x['note'])
                    if mm:
                        got = resolve(int(mm.group(1)), mm.group(2), depth + 1)
                        if got and (best is None or got[0] < best[0]):
                            best = got
            return best
        direct = cg_rows(lb)
        if direct:
            kind = 'root' if any(x['method'] == 'cg_v2_root' for x in direct) else 'bendex'
            claim, via = lb, ''
        else:
            cands = [resolve(lb, m) for m in r['lb_methods'].split()]
            cands = [c for c in cands if c]
            if not cands:
                print('cannot resolve', k, r['lb_methods'])
                continue
            claim, meth = min(cands)
            kind = 'root' if any(x['method'] == 'cg_v2_root' for x in cg_rows(claim)) else 'bendex'
            via = f'final LB {lb} from {r["lb_methods"]} relies on cg LB {claim}'
        out.append(dict(name=r['name'], variant=r['variant'], n=r['n'], claimed_lb=claim, final_lb=lb,
                        ub=r['ub'], closed=int(r['closed']) and claim == int(r['ub']) and 1 or 0,
                        table_closed=r['closed'], kind=kind, via=via))
    out.sort(key=lambda d: (-int(d['table_closed']), int(d['n']), d['variant'], d['name']))
    os.makedirs(OUT, exist_ok=True)
    with open(CLAIMS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(len(out), 'claims;', collections.Counter((d['kind'], d['table_closed']) for d in out))


def load_claims():
    return list(csv.DictReader(open(CLAIMS)))


# ----------------------------------------------------------------------------- instance
class Inst:
    """distinct raw bin lines t = (W, H, c) (dominated ones too: nothing relies on dominance)."""

    def __init__(self, nm, variant):
        bins, items = io.read(nm, variant == 'area')
        self.nm, self.variant = nm, variant
        self.items = [(int(w), int(h)) for (w, h) in items]
        self.types = sorted(set((int(W), int(H), int(c)) for (W, H, c) in bins))
        self.n, self.T = len(self.items), len(self.types)
        self.fit = [[w <= W and h <= H for (W, H, c) in self.types] for (w, h) in self.items]
        assert all(any(f) for f in self.fit), 'an item fits in no bin type'
        assert all(c > 0 for (W, H, c) in self.types)
        # an optimal packing has no empty bin (costs > 0), so n_t <= #items fitting t
        self.nmax = [sum(1 for f in self.fit if f[t]) for t in range(self.T)]


# ----------------------------------------------------------------------------- knapsack bounds P_t
def dp_bound(W, H, items, wts):
    """exact max of w(S) over S with area(S) <= W*H and at most one item 'big' in W x H (an upper bound of
    max w(S) over S 2D-packable in W x H).  Integer arithmetic (int64; values here are < 2^47)."""
    cap = W * H
    dp = np.zeros(cap + 1, dtype=np.int64)
    big = []
    for (w, h), v in zip(items, wts):
        v = int(v)
        if v <= 0 or w > W or h > H:
            continue
        if 2 * w > W and 2 * h > H:
            big.append((w * h, v))
            continue
        a = w * h
        dp[a:] = np.maximum(dp[a:], dp[:cap + 1 - a] + v)   # RHS built from the old dp: 0/1 knapsack
    best = int(dp[cap])
    for a, v in big:
        best = max(best, v + int(dp[cap - a]))
    return best


def cp_knap(W, H, items, wts, tl, workers, target=None):
    """independent CP-SAT 2D knapsack.  Returns (P, how) with P a proven upper bound of max w(S) over S
    2D-packable in W x H (no rotation), or (None, how).  With target K the model requires w(S) >= K+1:
    INFEASIBLE proves max w(S) <= K."""
    cand = [i for i, (w, h) in enumerate(items) if w <= W and h <= H and int(wts[i]) > 0]
    if not cand:
        return 0, 'empty'
    m = cp_model.CpModel()
    z = []; xi = []; yi = []
    for i in cand:
        w, h = items[i]
        b = m.NewBoolVar(''); x = m.NewIntVar(0, W - w, ''); y = m.NewIntVar(0, H - h, '')
        xi.append(m.NewOptionalFixedSizeIntervalVar(x, w, b, ''))
        yi.append(m.NewOptionalFixedSizeIntervalVar(y, h, b, ''))
        z.append(b)
    m.AddNoOverlap2D(xi, yi)
    # redundant (valid for any non-overlapping placement inside the bin)
    m.Add(sum(items[i][0] * items[i][1] * b for i, b in zip(cand, z)) <= W * H)
    m.AddCumulative(xi, [items[i][1] for i in cand], H)
    m.AddCumulative(yi, [items[i][0] for i in cand], W)
    obj = sum(int(wts[i]) * b for i, b in zip(cand, z))
    if target is not None:
        m.Add(obj >= int(target) + 1)
    m.Maximize(obj)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    st = s.Solve(m)
    if st == cp_model.INFEASIBLE:
        if target is None:
            return None, 'infeasible?!'      # cannot happen (S = {} is feasible)
        return int(target), 'tgt-infeasible'
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bb = s.BestObjectiveBound(); ov = s.ObjectiveValue()
        if not math.isfinite(bb):
            return None, 'nobound'
        P = max(int(math.ceil(bb)), int(round(ov)))
        if target is not None:
            P = max(P, int(target))
        return P, ('optimal' if st == cp_model.OPTIMAL else 'bound') + ('>tgt' if target is not None else '')
    return None, 'unknown'


# ----------------------------------------------------------------------------- combinatorial constraints
def best_clique(I, cand, Q, tries=40, seed=0):
    """a large set of items of cand pairwise conflicting in every type of Q in which both fit (greedy)."""
    rng = np.random.default_rng(seed)
    items, types, fit = I.items, I.types, I.fit

    def conf(i, j):
        (wi, hi), (wj, hj) = items[i], items[j]
        return all(wi + wj > types[t][0] and hi + hj > types[t][1] for t in Q if fit[i][t] and fit[j][t])
    adj = {i: set(j for j in cand if j != i and conf(i, j)) for i in cand}
    orders = [sorted(cand, key=lambda i: -items[i][0] * items[i][1]), sorted(cand, key=lambda i: -items[i][0]),
              sorted(cand, key=lambda i: -items[i][1]), sorted(cand, key=lambda i: -len(adj[i]))]
    for _ in range(tries):
        orders.append(list(rng.permutation(cand)))
    best = []
    for o in orders:
        K = []
        for i in o:
            if all(j in adj[i] for j in K):
                K.append(i)
        if len(K) > len(best):
            best = K
    # sanity: re-check pairwise
    assert all(conf(i, j) for a, i in enumerate(best) for j in best[a + 1:])
    return best


def comb_constraints(I):
    """list of (coef, rhs, label) meaning sum_t coef_t n_t >= rhs; see module docstring (A), (K)."""
    T = I.T
    out = []
    for r in range(1, T + 1):
        for Q in combinations(range(T), r):
            only = [i for i, f in enumerate(I.fit) if not any(f[t] for t in range(T) if t not in Q)]
            if not only:
                continue
            aQ = sum(I.items[i][0] * I.items[i][1] for i in only)
            out.append(([I.types[t][0] * I.types[t][1] if t in Q else 0 for t in range(T)], aQ, f'area{Q}'))
            K = best_clique(I, only, Q)
            out.append(([1 if t in Q else 0 for t in range(T)], len(K), f'clique{Q}'))
    return out


# ----------------------------------------------------------------------------- type-count IP
def solve_ip(I, cons, nogoods, tl=60.0, workers=1):
    """min sum c_t n_t over the box, cons (covering rows), no-goods (not n <= ng).  (val, n*) or (None, status)."""
    T = I.T
    m = cp_model.CpModel()
    nv = [m.NewIntVar(0, I.nmax[t], f'n{t}') for t in range(T)]
    for coef, rhs, lab in cons:
        if rhs <= 0:
            continue
        terms = [(int(coef[t]), nv[t]) for t in range(T) if coef[t]]
        if not terms:
            return None, f'trivially infeasible row {lab}'
        m.Add(sum(a * v for a, v in terms) >= int(rhs))
    for ng in nogoods:
        bs = []
        for t in range(T):
            if ng[t] + 1 <= I.nmax[t]:
                b = m.NewBoolVar('')
                m.Add(nv[t] >= ng[t] + 1).OnlyEnforceIf(b)
                bs.append(b)
        if not bs:
            return None, 'no-good excludes the whole box'
        m.AddBoolOr(bs)
    m.Minimize(sum(c * nv[t] for t, (W, H, c) in enumerate(I.types)))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
    st = s.Solve(m)
    if st != cp_model.OPTIMAL:
        return None, s.StatusName(st)
    nstar = [int(s.Value(v)) for v in nv]
    val = sum(I.types[t][2] * nstar[t] for t in range(T))
    assert feasible(I, cons, nogoods, nstar), 'CP-SAT returned an infeasible count vector'
    return val, nstar


def feasible(I, cons, nogoods, n):
    if any(n[t] < 0 or n[t] > I.nmax[t] for t in range(I.T)):
        return False
    if any(sum(int(a) * x for a, x in zip(coef, n)) < rhs for coef, rhs, _ in cons):
        return False
    return not any(all(n[t] <= ng[t] for t in range(I.T)) for ng in nogoods)


def enum_check(I, cons, nogoods, val, cap=4_000_000):
    """exhaustive integer check, no CP-SAT: True if no count vector of cost <= val-1 in the box satisfies all
    rows and no-goods; a counterexample vector if one does; None if the enumeration is too large."""
    T = I.T; c = [t[2] for t in I.types]; budget = val - 1
    if budget < 0:
        return True
    order = sorted(range(T), key=lambda t: -c[t])
    last = order[-1]
    pre = np.zeros((1, T), dtype=np.int64); cost = np.zeros(1, dtype=np.int64)
    for t in order[:-1]:
        kmax = np.minimum(I.nmax[t], (budget - cost) // c[t])
        reps = kmax + 1
        if int(reps.sum()) > cap:
            return None
        idx = np.repeat(np.arange(len(pre)), reps)
        ks = np.concatenate([np.arange(k + 1) for k in kmax]) if len(kmax) else np.zeros(0, dtype=np.int64)
        pre = pre[idx].copy(); pre[:, t] = ks; cost = cost[idx] + ks * c[t]
    need = np.zeros(len(pre), dtype=np.int64); bad = np.zeros(len(pre), dtype=bool)
    for coef, rhs, _ in cons:
        if rhs <= 0:
            continue
        assert max(int(a) for a in coef) * max(I.nmax) * T < 2 ** 62 and int(rhs) < 2 ** 62   # int64 is exact
        lhs = np.zeros(len(pre), dtype=np.int64)
        for t in order[:-1]:
            if coef[t]:
                lhs = lhs + pre[:, t] * int(coef[t])
        r = int(rhs) - lhs
        if coef[last] == 0:
            bad |= np.asarray(r > 0, dtype=bool)
        else:
            q = -((-r) // int(coef[last]))
            need = np.maximum(need, np.asarray(np.maximum(q, 0), dtype=np.int64))
    for ng in nogoods:
        dom = np.all(pre[:, [t for t in order[:-1]]] <= np.array([ng[t] for t in order[:-1]]), axis=1)
        need = np.where(dom, np.maximum(need, ng[last] + 1), need)
    ok = (~bad) & (need <= I.nmax[last]) & (cost + need * c[last] <= budget)
    if ok.any():
        k = int(np.nonzero(ok)[0][0]); v = [int(x) for x in pre[k]]; v[last] = int(need[k])
        return v
    return True


# ----------------------------------------------------------------------------- fixed-count feasibility (no-goods)
def fixed_slots(I, nvec, tl, workers):
    """can all items be packed into <= nvec[t] bins of each type t?  'infeasible' / 'feasible' / 'unknown'.
    Own model: one slot per bin copy, NoOverlap2D + area + cumulatives per slot, each item in exactly one slot;
    symmetry breaking: rank = (area desc, index); slot k of a type only holds items of rank >= k."""
    items = I.items
    order = sorted(range(I.n), key=lambda i: (-items[i][0] * items[i][1], i))
    rank = {i: q for q, i in enumerate(order)}
    m = cp_model.CpModel(); asg = [[] for _ in items]; slots = []
    for t, (W, H, c) in enumerate(I.types):
        for k in range(nvec[t]):
            xi = []; yi = []; recs = []
            for i, (w, h) in enumerate(items):
                if w > W or h > H or rank[i] < k:
                    continue
                a = m.NewBoolVar(''); x = m.NewIntVar(0, W - w, ''); y = m.NewIntVar(0, H - h, '')
                xi.append(m.NewOptionalFixedSizeIntervalVar(x, w, a, ''))
                yi.append(m.NewOptionalFixedSizeIntervalVar(y, h, a, ''))
                asg[i].append(a); recs.append((i, a, x, y))
            if recs:
                m.AddNoOverlap2D(xi, yi)
                m.Add(sum(items[i][0] * items[i][1] * a for (i, a, _, _) in recs) <= W * H)
                m.AddCumulative(xi, [items[i][1] for (i, _, _, _) in recs], H)
                m.AddCumulative(yi, [items[i][0] for (i, _, _, _) in recs], W)
            slots.append((t, k, recs))
    if any(not a for a in asg):
        return 'infeasible', 'an item has no slot'
    for i in range(I.n):
        m.AddExactlyOne(asg[i])
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
    st = s.Solve(m)
    if st == cp_model.INFEASIBLE:
        return 'infeasible', round(s.WallTime(), 1)
    if st in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        place = [[t, k, items[i][0], items[i][1], s.Value(x), s.Value(y)]
                 for (t, k, recs) in slots for (i, a, x, y) in recs if s.Value(a)]
        return 'feasible', place
    return 'unknown', round(s.WallTime(), 1)


def solution_check(I, vecs, rows, ngs):
    """consistency test on the table's checked packing (results/final/solutions): every bin's item set must
    respect every P_t, and its count vector must satisfy every row and no-good (it is a feasible packing).
    Identical rectangles are interchangeable, so items are matched to the placements by dimensions."""
    f = os.path.join(ROOT, 'results', 'final', 'solutions', f'{I.nm}_{I.variant}.json')
    if not os.path.exists(f):
        return 'no packing file'
    raw, _ = io.read(I.nm, I.variant == 'area')
    place = json.load(open(f))['place']
    pool = collections.defaultdict(list)
    for i, wh in enumerate(I.items):
        pool[wh].append(i)
    bins = collections.defaultdict(list)
    for r, k, w, h, x, y in place:
        W, H, c = raw[r]
        if not (x >= 0 and y >= 0 and x + w <= W and y + h <= H):
            return 'packing file invalid (outside)'
        bins[(r, k)].append((pool[(w, h)].pop(), x, y))
    if any(pool.values()):
        return 'packing file invalid (items missing)'
    nvec = [0] * I.T
    for (r, k), S in bins.items():
        t = I.types.index(tuple(int(v) for v in raw[r]))
        nvec[t] += 1
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                (i, xa, ya), (j, xb, yb) = S[a], S[b]
                (wa, ha), (wb, hb) = I.items[i], I.items[j]
                if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                    return 'packing file invalid (overlap)'
        for v in vecs:
            ws = sum(v['wts'][i] for i, _, _ in S)
            if ws > v['P'][t]:
                return f"VIOLATION: a packed bin of type {I.types[t]} has weight {ws} > P_t {v['P'][t]} ({v['src']}, {v['how'][t]})"
    for coef, rhs, lab in rows:
        if sum(int(a) * x for a, x in zip(coef, nvec)) < rhs:
            return f'VIOLATION: checked packing with counts {nvec} violates row {lab}'
    for ng in ngs:
        if all(nvec[t] <= ng[t] for t in range(I.T)):
            return f'VIOLATION: checked packing with counts {nvec} violates no-good {list(ng)}'
    return f'ok (counts {nvec})'


# ----------------------------------------------------------------------------- weight vectors
def jsonl_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def certs_for(nm, variant):
    return sorted(glob.glob(os.path.join(ROOT, 'results', 'cg', 'certs', f'{nm}_{variant}_bendex_*.json')))


def collect_vectors(I):
    """weight vectors: dict(src, wts (python ints), opt (per-type optimistic coefficient, i.e. the value a
    cut is expected to need), certcoef (per-type coefficient claimed by cg_bendex, for the report))."""
    vecs = []; seen = set()

    def add(src, wts, opt, certcoef=None):
        key = tuple(wts)
        if not any(wts) or key in seen:
            return
        seen.add(key)
        vecs.append(dict(src=src, wts=list(wts), opt=list(opt), certcoef=certcoef))

    add('areas', [w * h for (w, h) in I.items], [W * H for (W, H, c) in I.types])

    def add_pi(src, pi):
        if pi is None:
            return
        pi = [max(0.0, float(p)) for p in pi]
        mx = max(pi)
        if len(pi) != I.n or mx <= 0:
            return
        s = int(1e9 / mx)
        wts = [int(math.floor(p * s)) for p in pi]
        # optimistic coefficient: reduced costs >= 0 at LP-optimal duals, w(S) <= s*pi(S) <= s*c_t (+ GLOP tol)
        add(src, wts, [int(math.floor(c * s * (1 + 1e-8))) for (W, H, c) in I.types])
    srcs = []
    for f in [os.path.join(OUT, 'rerun.jsonl')] + sorted(glob.glob(os.path.join(ROOT, 'results', 'cg', '*.jsonl'))):
        for d in jsonl_rows(f):
            if d.get('name') == I.nm and d.get('variant') == I.variant:
                srcs.append((os.path.basename(f), d))
    certs = [(os.path.basename(f), json.load(open(f))) for f in certs_for(I.nm, I.variant)]
    for f, d in srcs + certs:
        add_pi(f + ':piB', d.get('piB'))
        add_pi(f + ':piA', d.get('piA'))
    ttuple = [tuple(t) for t in I.types]
    for f, d in certs:
        ct = [tuple(t) for t in d['types']]
        if sorted(ct) != ttuple:
            print('  !! cert types differ', f, flush=True)
            continue
        for k, cut in enumerate(d['cuts']):
            if cut.get('src') != 'benders' or not cut.get('wts'):
                continue        # 'initial' cuts come from the areas / piA / piB used above
            wts = [int(v) for v in cut['wts']]
            if len(wts) != I.n or min(wts) < 0:
                continue
            cc = {ct[t]: int(cut['coef'][t]) for t in range(len(ct))}
            coef = [cc[t] for t in ttuple]
            add(f'{f}:cut{k}', wts, coef, certcoef=coef)
    return vecs


def cert_nogoods(I):
    out = []; seen = set()
    ttuple = [tuple(t) for t in I.types]
    for f in certs_for(I.nm, I.variant):
        d = json.load(open(f))
        ct = [tuple(t) for t in d['types']]
        if sorted(ct) != ttuple:
            continue
        for ng in d.get('nogoods', []):
            m = {ct[t]: int(ng[t]) for t in range(len(ct))}
            v = tuple(m[t] for t in ttuple)
            if v not in seen:
                seen.add(v); out.append(v)
    return out


# ----------------------------------------------------------------------------- one claim
def verify_claim(d, tl=15.0, workers=1, budget=600.0, ng_tl=60.0, own_ng=0, log=print):
    t0 = time.time()
    nm, var, claim = d['name'], d['variant'], int(d['claimed_lb'])
    target = max(claim, int(d.get('target') or 0))    # stop when this is reached (>= claim)
    I = Inst(nm, var); T = I.T
    comb = comb_constraints(I)
    vecs = collect_vectors(I)
    for v in vecs:
        v['rhs'] = sum(v['wts'])
        v['P'] = [dp_bound(W, H, I.items, v['wts']) for (W, H, c) in I.types]
        v['how'] = ['dp'] * T
        v['done'] = [v['P'][t] <= v['opt'][t] for t in range(T)]   # DP already reaches the optimistic value
    ng_cand = cert_nogoods(I); ng_tried = set(); ngs = []; ng_src = []
    nk = 0; tk = 0.0; alerts = []; own = 0; it = 0
    hist = []

    def rows():
        return comb + [(v['P'], v['rhs'], v['src']) for v in vecs]
    while True:
        it += 1
        val, nstar = solve_ip(I, rows(), ngs, workers=workers)
        if val is None:
            alerts.append(f'IP not solved: {nstar}')
            break
        hist.append(val)
        if val >= target or time.time() - t0 > budget:
            break
        todo = []
        for v in vecs:
            opt = [v['P'][t] if v['done'][t] else min(v['P'][t], v['opt'][t]) for t in range(T)]
            if sum(opt[t] * nstar[t] for t in range(T)) < v['rhs']:
                todo += [(v, t) for t in range(T) if nstar[t] > 0 and not v['done'][t]]
        if todo:
            for v, t in todo:
                if time.time() - t0 > budget:
                    break
                W, H, c = I.types[t]
                ta = time.time()
                P, how = cp_knap(W, H, I.items, v['wts'], tl, workers, target=v['opt'][t])
                tk += time.time() - ta; nk += 1
                v['done'][t] = True
                if P is not None and P < v['P'][t]:
                    v['P'][t] = P; v['how'][t] = how
                else:
                    v['how'][t] = f'dp({how})'
                if v['certcoef'] is not None and how.endswith('>tgt'):
                    alerts.append(f"cg_bendex cut {v['src']} type {I.types[t]}: a packable set has weight > coef "
                                  f"{v['certcoef'][t]} (CP-SAT {how} {P}) - INVALID CUT")
            continue
        cands = [ng for ng in ng_cand if ng not in ng_tried and all(nstar[t] <= ng[t] for t in range(T))]
        if cands and time.time() - t0 < budget:
            ng = cands[0]; ng_tried.add(ng)
            verdict, info = fixed_slots(I, ng, ng_tl, workers)
            log(f'    cert no-good {list(ng)}: {verdict}')
            if verdict == 'infeasible':
                ngs.append(ng); ng_src.append('cert')
            elif verdict == 'feasible':
                alerts.append(f'cg_bendex no-good {list(ng)} REFUTED: items fit into these counts')
            continue
        if own < own_ng and time.time() - t0 < budget:
            own += 1
            verdict, info = fixed_slots(I, nstar, ng_tl, workers)
            log(f'    own no-good check {nstar}: {verdict}')
            if verdict == 'infeasible':
                ngs.append(tuple(nstar)); ng_src.append('own')
                continue
            if verdict == 'feasible':
                alerts.append(f'items fit into counts {nstar} of cost {val} < claim {claim}: CLAIM REFUTED')
        break
    res = dict(name=nm, variant=var, claimed_lb=claim, target=target, safe_lb=val, nstar=nstar if val is not None else None,
               iters=it, hist=hist, n_vecs=len(vecs), knap_calls=nk, knap_time=round(tk, 1),
               nogoods_cert=sum(1 for s in ng_src if s == 'cert'), nogoods_cert_avail=len(ng_cand),
               nogoods_own=sum(1 for s in ng_src if s == 'own'), alerts=alerts)
    if val is not None:
        e = enum_check(I, rows(), ngs, val)
        res['enum'] = 'ok' if e is True else 'too-large' if e is None else f'COUNTEREXAMPLE {e}'
        if e is not True and e is not None:
            alerts.append(f'enumeration found a count vector {e} cheaper than CP-SAT value {val}')
            res['safe_lb'] = val = None
    res['solcheck'] = solution_check(I, vecs, rows(), ngs)
    if res['solcheck'].startswith('VIOLATION'):
        alerts.append(res['solcheck'])
        res['safe_lb'] = val = None
    # which coefficients needed CP-SAT (below the DP value)?
    res['cp_coefs'] = sum(1 for v in vecs for t in range(T) if not v['how'][t].startswith('dp'))
    res['binding'] = [] if val is None else [
        v['src'] for v in vecs if sum(v['P'][t] * res['nstar'][t] for t in range(T)) < v['rhs'] + max(v['P'])]
    res['status'] = ('FAILED' if val is None or any('REFUTED' in a or 'INVALID' in a for a in alerts)
                     else 'SAFE' if val >= claim else 'WEAKER')
    res['time'] = round(time.time() - t0, 1)
    res['threads'] = workers
    return res


def append_jsonl(path, obj):
    line = (json.dumps(obj) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def ready(d, start_ts):
    """inputs of the re-runs available for this claim?"""
    nm, var, kind = d['name'], d['variant'], d['kind']
    f = os.path.join(OUT, 'rerun.jsonl' if kind == 'root' else 'bendex.jsonl')
    if any(r.get('name') == nm and r.get('variant') == var for r in jsonl_rows(f)):
        return True
    td = os.path.join(OUT, 'rerun_tasks', f'{kind}_{var}_{nm}')
    return os.path.isdir(td) and time.time() - os.path.getmtime(td) > (300 if kind == 'root' else 900)


def rerun_lb(d):
    """best LB reported by the re-runs made for this check (cg_v2 lb_round / cg_bendex lb) for the claim's pair."""
    best = 0
    for f, key in (('rerun.jsonl', 'lb_round'), ('bendex.jsonl', 'lb')):
        for r in jsonl_rows(os.path.join(OUT, f)):
            if r.get('name') == d['name'] and r.get('variant') == d['variant'] and r.get(key) is not None:
                if r.get('best_checked_ub') is None or r[key] <= r['best_checked_ub']:
                    best = max(best, int(r[key]))
    return best


def run_verify(a):
    claims = load_claims()
    if a.names:
        claims = [d for d in claims if d['name'] in a.names]
    if a.closed in ('0', '1'):
        claims = [d for d in claims if d['table_closed'] == a.closed]
    if a.kind:
        claims = [d for d in claims if d['kind'] == a.kind]
    tdir = os.path.join(OUT, 'verify_tasks' + (a.tag or ''))
    os.makedirs(tdir, exist_ok=True)
    stop_ts = None
    if a.stop:
        hh, mm = map(int, a.stop.split(':'))
        lt = time.localtime(); stop_ts = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    pending = list(claims)
    while pending:
        if stop_ts and time.time() > stop_ts:
            print('stop time reached', flush=True); break
        nxt = []
        progressed = False
        for d in pending:
            if stop_ts and time.time() > stop_ts:
                break
            if not a.no_wait and not ready(d, 0):
                nxt.append(d); continue
            if a.target_rerun:
                d = dict(d, target=rerun_lb(d))
            try:
                os.mkdir(os.path.join(tdir, f"{d['variant']}_{d['name']}"))
            except FileExistsError:
                continue
            progressed = True
            print(f"=== {time.strftime('%H:%M:%S')} {d['name']} {d['variant']} claim {d['claimed_lb']} ({d['kind']}, closed={d['table_closed']})", flush=True)
            try:
                res = verify_claim(d, tl=a.tl, workers=a.workers, budget=a.budget, ng_tl=a.ng_tl, own_ng=a.own_ng,
                                   log=lambda s: print(s, flush=True))
            except Exception as e:  # report, never hide
                import traceback; traceback.print_exc()
                res = dict(name=d['name'], variant=d['variant'], claimed_lb=int(d['claimed_lb']), safe_lb=None,
                           status='FAILED', alerts=[f'exception {e!r}'], time=0)
            res['kind'] = d['kind']; res['closed'] = int(d['table_closed']); res['tag'] = a.tag or ''
            res['ts'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            print('   ', res['status'], res.get('safe_lb'), 'vs', res['claimed_lb'], f"{res['time']}s",
                  {k: res.get(k) for k in ('knap_calls', 'nogoods_cert', 'nogoods_own', 'enum', 'alerts')}, flush=True)
            append_jsonl(VERIFY, res)
        pending = nxt
        if pending and not progressed:
            time.sleep(30)
    print('verify done', flush=True)


# ----------------------------------------------------------------------------- summary
def best_results():
    best = {}
    for r in jsonl_rows(VERIFY):
        k = (r['name'], r['variant'])
        rank = (r.get('status') != 'FAILED', r.get('safe_lb') or -1, 'solcheck' in r, r.get('ts', ''))
        if k not in best or rank > best[k][0]:
            best[k] = (rank, r)
    return {k: v[1] for k, v in best.items()}


def summary(record=False):
    claims = load_claims()
    B = best_results()
    rows = []
    for d in claims:
        k = (d['name'], d['variant'])
        r = B.get(k)
        if r is None:
            rows.append(dict(name=d['name'], variant=d['variant'], claimed_lb=d['claimed_lb'], safe_lb='',
                             status='NOT_RUN', time='', closed=d['table_closed'], kind=d['kind'], notes=d['via']))
            continue
        notes = []
        if d['via']:
            notes.append(d['via'])
        if r.get('status') != 'FAILED' or r.get('safe_lb') is not None:
            notes.append(f"n*={r.get('nstar')} vecs={r.get('n_vecs')} cp_knap={r.get('knap_calls')} "
                         f"cp_coefs={r.get('cp_coefs')} nogoods(cert {r.get('nogoods_cert')}/{r.get('nogoods_cert_avail')} "
                         f"own {r.get('nogoods_own')}) enum={r.get('enum')} packing-check={r.get('solcheck', 'n/a')}")
        notes += r.get('alerts', [])
        rows.append(dict(name=d['name'], variant=d['variant'], claimed_lb=d['claimed_lb'],
                         safe_lb='' if r.get('safe_lb') is None else r['safe_lb'], status=r['status'],
                         time=r.get('time'), closed=d['table_closed'], kind=d['kind'], notes=' | '.join(notes)))
    with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    cnt = collections.Counter((r['closed'], r['status']) for r in rows)
    print('closed=1:', {s: cnt[('1', s)] for s in ('SAFE', 'WEAKER', 'FAILED', 'NOT_RUN')})
    print('closed=0:', {s: cnt[('0', s)] for s in ('SAFE', 'WEAKER', 'FAILED', 'NOT_RUN')})
    for r in rows:
        if r['status'] in ('WEAKER', 'FAILED'):
            print(r['name'], r['variant'], 'closed' if r['closed'] == '1' else 'open', r['claimed_lb'], r['safe_lb'],
                  r['status'], r['notes'][-200:])
    if record:
        have = set((r['name'], r['variant'], r['lb']) for r in ledger.rows() if r['method'] == 'verify_cg_safe')
        best = ledger.best()
        nrec = 0
        for r in rows:
            if r['status'] != 'SAFE':
                continue
            lb = int(r['safe_lb'])
            if (r['name'], r['variant'], str(lb)) in have:
                continue
            ub = best.get((r['name'], r['variant']), {}).get('ub')
            if ub is not None and lb > ub:
                print('!!! safe LB above checked UB, not recorded', r['name'], r['variant'], lb, ub)
                continue
            ledger.record(r['name'], r['variant'], 'verify_cg_safe', 'LB_ONLY', lb=lb, time_s=r['time'], threads=1,
                          note=f"integer re-derivation of cg LB {r['claimed_lb']} (exact knapsack bounds; "
                               f"re-proven no-goods; CP-SAT type-count IP; enum={B[(r['name'], r['variant'])].get('enum')})")
            nrec += 1
        print('recorded', nrec, 'ledger rows')
    return rows


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'names':
        kind, var = sys.argv[2], sys.argv[3]
        cl = sys.argv[4] if len(sys.argv) > 4 else None
        print(' '.join(d['name'] for d in load_claims() if d['kind'] == kind and d['variant'] == var
                       and (cl is None or d['table_closed'] == cl)))
        return
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['claims', 'verify', 'summary'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--tl', type=float, default=15.0, help='CP-SAT time per 2D knapsack bound')
    ap.add_argument('--ng-tl', type=float, default=60.0, help='CP-SAT time per fixed-count no-good check')
    ap.add_argument('--budget', type=float, default=600.0, help='time per claim')
    ap.add_argument('--own-ng', type=int, default=0, help='max own fixed-count no-goods per claim')
    ap.add_argument('--closed', default='all')
    ap.add_argument('--kind', default=None)
    ap.add_argument('--stop', default=None, help='local HH:MM: no new claim after it')
    ap.add_argument('--tag', default='', help='suffix of the task-claim directory (use a new tag for a second pass)')
    ap.add_argument('--no-wait', action='store_true', help='do not wait for re-run outputs')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--target-rerun', action='store_true', help='aim for the (higher) LB of the re-runs, not just the claim')
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    if a.cmd == 'claims':
        build_claims()
    elif a.cmd == 'verify':
        run_verify(a)
    else:
        summary(a.record)


if __name__ == '__main__':
    main()
