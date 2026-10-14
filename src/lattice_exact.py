"""Cost-lattice enumeration + fixed-bin-multiset 2D packing feasibility ('lattice_opp').

Exact method for the 2DVSBPP (oriented items, non-guillotine, unlimited supply per bin type).

Idea
  The objective only takes values on the lattice {sum_t c_t n_t}. Enumerate bin-count vectors
  n = (n_t) (over NON-DOMINATED types, see reduce_types) with cost(n) <= U in non-decreasing cost
  order and decide for each one whether the items can be packed into exactly that multiset of
  bins with EVERY bin non-empty:
    (a) cheap necessary-condition filters (pure Python):
          - every item fits some chosen type;
          - Hall/transportation area condition: for every subset S of the chosen types, the items
            that fit ONLY into types of S have total area <= sum_{t in S} n_t W_t H_t;
          - conflict cliques: items that pairwise cannot share any chosen bin (also restricted
            to the items that fit only into a subset S) need distinct bins;
          - number of bins <= number of items (non-empty bins);
    (b) a combinatorial-Benders loop: master = CP-SAT assignment relaxation (area, 'tall'/'wide'
        width/height sums, conflict at-most-ones, non-empty bins, symmetry breaking), sub-problem =
        per-bin 2D orthogonal packing (CP-SAT no_overlap_2d). A sub-problem proven INFEASIBLE
        yields a no-good, lifted to every chosen bin type no larger in both dimensions;
    (c) if (b) neither refutes nor packs within its budget: one full CP-SAT feasibility model with
        the fixed bins (assignment + no_overlap_2d per bin + area/cumulative redundancy + the
        relaxation constraints + symmetry breaking), no objective, under a time limit.
  A vector is REFUTED only by (a), or by CP-SAT returning INFEASIBLE in (b)/(c); a time-out
  (UNKNOWN) never refutes. The first vector found feasible (in cost order) is optimal if every
  cheaper vector was refuted. LB = cost of the cheapest vector that is not refuted.

Validity notes (why each LB is a proof)
  * Dominated types: if type t has W_t >= W_s, H_t >= H_s, c_t <= c_s, every s-bin of a packing
    can be replaced by a t-bin (same item positions) without increasing cost, so some optimal
    packing uses non-dominated types only. Identical lines are deduplicated (lowest raw index kept).
  * Non-empty bins: a packing with an empty bin can drop it and get strictly cheaper (c_t > 0).
    So the optimum is attained by a packing whose bins are all non-empty, and whose vector uses
    non-dominated types; it has n_t <= #items fitting in t and sum n_t <= n; it has cost <= U if
    U is achievable. Hence: if every vector with these properties and cost <= V is refuted
    (with the non-empty requirement), OPT > V.
  * Symmetry breaking among the n_t identical copies of a type: relabel the copies by increasing
    smallest item index (items sorted by non-increasing area, ties arbitrary but fixed). With
    non-empty bins the minima are distinct, so 'min index of copy k < min index of copy k+1'
    is satisfied by the relabelled packing.
  * Tall/wide cuts: two items with h_i + h_j > H cannot be stacked, so items with h > H/2 in one
    bin occupy disjoint x-ranges: sum of their widths <= W (symmetric for w > W/2).
  * Conflicts: i, j cannot share a t-bin if they do not both fit, or w_i + w_j > W and
    h_i + h_j > H (then any two placements overlap in both projections).
  * Benders no-goods: only from a CP-SAT INFEASIBLE proof that item set S does not fit into
    W x H; then S does not fit into any W' x H' with W' <= W, H' <= H.

usage:
  python src/lattice_exact.py --variant area --time 600 --vec-time 60 MB_C8_1
  options: --workers 2, --ub N (achievable UB; default min(ledger checked UB, CSV BKS)),
           --use-ledger-lb (skip vectors with cost < best ledger LB; the LB then relies on it),
           --continue (keep going after an UNKNOWN vector to look for a better UB),
           --no-record (do not write to the ledger)
"""
import argparse
import json
import os
import subprocess
import sys
import time

from ortools.sat.python import cp_model

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vsbpp_io as io  # noqa: E402
import ledger  # noqa: E402

ROOT = os.path.dirname(HERE)
SOL_DIR = os.path.join(ROOT, 'results', 'solutions', 'sprint')
METHOD = 'lattice_opp'


# --------------------------------------------------------------------------- instance helpers
def reduce_types(bins_raw):
    """Non-dominated, deduplicated bin types: list of (W, H, c, raw_index).

    s is dropped if some other line t has W_t >= W_s, H_t >= H_s, c_t <= c_s (for identical
    lines the lower raw index is kept). Valid for LB and UB, see module docstring."""
    keep = []
    for s, (Ws, Hs, cs) in enumerate(bins_raw):
        dom = False
        for t, (Wt, Ht, ct) in enumerate(bins_raw):
            if t == s:
                continue
            if Wt >= Ws and Ht >= Hs and ct <= cs:
                if (Wt, Ht, ct) != (Ws, Hs, cs) or t < s:
                    dom = True
                    break
        if not dom:
            keep.append((Ws, Hs, cs, s))
    return keep


def fits(it, ty):
    return it[0] <= ty[0] and it[1] <= ty[1]


def can_share(a, b, ty):
    """items a, b can be together in one bin of type ty (necessary and sufficient for 2 items)."""
    if not (fits(a, ty) and fits(b, ty)):
        return False
    return a[0] + b[0] <= ty[0] or a[1] + b[1] <= ty[1]


def enumerate_vectors(types, items, vmax):
    """All n with cost <= vmax, 0 <= n_t <= #items fitting t, sum n_t <= n. Sorted by cost."""
    n = len(items)
    caps = [min(sum(1 for it in items if fits(it, ty)), vmax // ty[2], n) for ty in types]
    out = []
    T = len(types)

    def dfs(t, cost, cnt, vec):
        if t == T:
            if cnt > 0:
                out.append((cost, tuple(vec)))
            return
        c = types[t][2]
        k = 0
        while k <= caps[t] and cost + k * c <= vmax and cnt + k <= n:
            vec.append(k)
            dfs(t + 1, cost + k * c, cnt + k, vec)
            vec.pop()
            k += 1
    dfs(0, 0, 0, [])
    out.sort()
    return out


# --------------------------------------------------------------------------- cheap filters
def greedy_clique(cands, conflict):
    """greedy clique in the conflict graph restricted to cands (list of item indices)."""
    best = 0
    order = sorted(cands, key=lambda i: -sum(conflict[i][j] for j in cands))
    for start in order[:min(len(order), 12)]:
        cl = [start]
        for j in order:
            if j != start and all(conflict[j][k] for k in cl):
                cl.append(j)
        best = max(best, len(cl))
    return best


def cheap_filter(types, items, vec):
    """return None if the vector survives, else a short reason (each is a necessary condition)."""
    n = len(items)
    used = [t for t in range(len(types)) if vec[t] > 0]
    if sum(vec) > n:
        return 'too_many_bins'
    fitset = []
    for it in items:
        f = frozenset(t for t in used if fits(it, types[t]))
        if not f:
            return 'item_fits_no_bin'
        fitset.append(f)
    area = [w * h for w, h in items]
    if sum(area) > sum(vec[t] * types[t][0] * types[t][1] for t in used):
        return 'area'
    # Hall-type area condition over subsets of used types
    U = len(used)
    for mask in range(1, 1 << U):
        S = frozenset(used[b] for b in range(U) if mask >> b & 1)
        cap = sum(vec[t] * types[t][0] * types[t][1] for t in S)
        need = sum(area[i] for i in range(n) if fitset[i] <= S)
        if need > cap:
            return 'hall_area'
    # conflict cliques (items pairwise unable to share any used bin type)
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = not any(can_share(items[i], items[j], types[t]) for t in used)
            conflict[i][j] = conflict[j][i] = c
    for mask in range(1, 1 << U):
        S = frozenset(used[b] for b in range(U) if mask >> b & 1)
        cands = [i for i in range(n) if fitset[i] <= S]
        if not cands:
            continue
        nb = sum(vec[t] for t in S)
        if len(cands) > nb and greedy_clique(cands, conflict) > nb:
            return 'clique'
    return None


# --------------------------------------------------------------------------- CP-SAT pieces
def normal_positions(sizes, cap):
    """Herz / Christofides-Whitlock normal patterns: all subset sums of `sizes` that are <= cap.
    Any feasible packing can be made bottom-left justified by repeatedly moving items down/left
    (this terminates); then every coordinate of an item is a sum of sizes of OTHER items. Moving
    items only down/left keeps 'largest item in the lower-left quadrant' true, so this is
    compatible with the reflection symmetry breaking used in opp()."""
    if cap < 0:
        return [0]
    reach = 1
    mask = (1 << (cap + 1)) - 1
    for s in sizes:
        reach = (reach | (reach << s)) & mask
    return [v for v in range(cap + 1) if (reach >> v) & 1]


def opp(items_sub, W, H, tl, workers):
    """2D orthogonal packing of items_sub into W x H. returns ('FEASIBLE', [(x,y)]) /
    ('INFEASIBLE', None) / ('UNKNOWN', None)."""
    if sum(w * h for w, h in items_sub) > W * H:
        return 'INFEASIBLE', None
    if any(w > W or h > H for w, h in items_sub):
        return 'INFEASIBLE', None
    m = cp_model.CpModel()
    xs, ys, xi, yi = [], [], [], []
    for q, (w, h) in enumerate(items_sub):
        x = m.NewIntVarFromDomain(cp_model.Domain.FromValues(
            normal_positions([it[0] for r, it in enumerate(items_sub) if r != q], W - w)), '')
        y = m.NewIntVarFromDomain(cp_model.Domain.FromValues(
            normal_positions([it[1] for r, it in enumerate(items_sub) if r != q], H - h)), '')
        xs.append(x); ys.append(y)
        xi.append(m.NewFixedSizeIntervalVar(x, w, '')); yi.append(m.NewFixedSizeIntervalVar(y, h, ''))
    m.AddNoOverlap2D(xi, yi)
    m.AddCumulative(xi, [h for w, h in items_sub], H)
    m.AddCumulative(yi, [w for w, h in items_sub], W)
    # symmetry: first (largest) item in the lower-left quadrant (reflection symmetry of the bin)
    w0, h0 = items_sub[0]
    m.Add(2 * xs[0] <= W - w0); m.Add(2 * ys[0] <= H - h0)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
    st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 'FEASIBLE', [(s.Value(xs[k]), s.Value(ys[k])) for k in range(len(items_sub))]
    if st == cp_model.INFEASIBLE:
        return 'INFEASIBLE', None
    return 'UNKNOWN', None


class VecModel:
    """assignment model for a fixed bin multiset; geometry optional."""

    def __init__(self, types, items, vec, geometry, nogoods=()):
        self.types, self.items, self.vec = types, items, vec
        n = len(items)
        m = self.m = cp_model.CpModel()
        self.bins = [(t, k) for t in range(len(types)) for k in range(vec[t])]
        a = self.a = {}
        self.pos = {}
        for b, (t, k) in enumerate(self.bins):
            W, H, c, _ = types[t]
            fit = [i for i in range(n) if fits(items[i], types[t]) and i >= k]
            for i in fit:
                a[i, b] = m.NewBoolVar(f'a{i}_{b}')
            lits = [a[i, b] for i in fit]
            m.AddBoolOr(lits)  # non-empty bin
            m.Add(sum(items[i][0] * items[i][1] * a[i, b] for i in fit) <= W * H)
            tall = [i for i in fit if 2 * items[i][1] > H]
            if len(tall) > 1:
                m.Add(sum(items[i][0] * a[i, b] for i in tall) <= W)
            wide = [i for i in fit if 2 * items[i][0] > W]
            if len(wide) > 1:
                m.Add(sum(items[i][1] * a[i, b] for i in wide) <= H)
            # conflict at-most-ones (greedy clique cover of pairs that cannot share type t)
            covered = set()
            for i in fit:
                cl = [i]
                for j in fit:
                    if j > i and all(not can_share(items[j], items[q], types[t]) for q in cl):
                        cl.append(j)
                if len(cl) > 1:
                    key = tuple(cl)
                    if key not in covered:
                        covered.add(key)
                        m.AddAtMostOne([a[q, b] for q in cl])
            if geometry:
                xi, yi = [], []
                for i in fit:
                    w, h = items[i]
                    # normal patterns w.r.t. ALL other items fitting this type (superset of the
                    # bin's content, hence still a valid restriction)
                    xv = m.NewIntVarFromDomain(cp_model.Domain.FromValues(
                        normal_positions([items[j][0] for j in fit if j != i], W - w)), '')
                    yv = m.NewIntVarFromDomain(cp_model.Domain.FromValues(
                        normal_positions([items[j][1] for j in fit if j != i], H - h)), '')
                    self.pos[i, b] = (xv, yv)
                    xi.append(m.NewOptionalFixedSizeIntervalVar(xv, w, a[i, b], ''))
                    yi.append(m.NewOptionalFixedSizeIntervalVar(yv, h, a[i, b], ''))
                m.AddNoOverlap2D(xi, yi)
                m.AddCumulative(xi, [items[i][1] for i in fit], H)
                m.AddCumulative(yi, [items[i][0] for i in fit], W)
        for i in range(n):
            m.AddExactlyOne([a[i, b] for b in range(len(self.bins)) if (i, b) in a])
        # symmetry breaking between consecutive copies of the same type:
        # min item index of copy k < min item index of copy k+1.
        # p[b][i] = bin b contains an item with index <= i
        for b, (t, k) in enumerate(self.bins):
            if k == 0:
                continue
            bp = b - 1  # previous copy of the same type
            assert self.bins[bp] == (t, k - 1)
            prev = None
            pref = []  # pref[i] = OR_{j<=i} a[j, bp]
            for i in range(n):
                if (i, bp) in a:
                    if prev is None:
                        cur = a[i, bp]
                    else:
                        cur = m.NewBoolVar('')
                        m.AddBoolOr([prev, a[i, bp]]).OnlyEnforceIf(cur)
                        m.AddImplication(prev, cur); m.AddImplication(a[i, bp], cur)
                    prev = cur
                pref.append(prev)
            for i in range(n):
                if (i, b) in a:
                    if i == 0 or pref[i - 1] is None:
                        m.Add(a[i, b] == 0)
                    else:
                        m.AddImplication(a[i, b], pref[i - 1])
        for S, (Wn, Hn) in nogoods:
            self.add_nogood(S, Wn, Hn)

    def add_nogood(self, S, Wn, Hn):
        """item set S proven not to fit Wn x Hn: forbid S in any chosen bin no larger."""
        for b, (t, k) in enumerate(self.bins):
            W, H = self.types[t][0], self.types[t][1]
            if W <= Wn and H <= Hn:
                lits = [self.a[i, b] for i in S if (i, b) in self.a]
                if len(lits) == len(S):
                    self.m.Add(sum(lits) <= len(S) - 1)

    def solve(self, tl, workers):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = max(0.1, tl)
        s.parameters.num_workers = workers
        st = s.Solve(self.m)
        self.solver = s
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return 'FEASIBLE'
        if st == cp_model.INFEASIBLE:
            return 'INFEASIBLE'
        return 'UNKNOWN'

    def assignment(self):
        s = self.solver
        out = [[] for _ in self.bins]
        for (i, b), v in self.a.items():
            if s.Value(v):
                out[b].append(i)
        return out

    def placement(self):
        s = self.solver
        res = []
        for (i, b), v in self.a.items():
            if s.Value(v):
                res.append((i, b, s.Value(self.pos[i, b][0]), s.Value(self.pos[i, b][1])))
        return res


def shrink_nogood(items, S, W, H, tl, workers, deadline):
    """deletion filter: drop items while the remaining set is still PROVEN not to fit."""
    S = sorted(S, key=lambda i: items[i][0] * items[i][1])  # try removing small items first
    k = 0
    while k < len(S) and time.time() < deadline:
        T = S[:k] + S[k + 1:]
        if len(T) >= 2:
            sub = sorted([items[i] for i in T], key=lambda it: -it[0] * it[1])
            st, _ = opp(sub, W, H, tl, workers)
            if st == 'INFEASIBLE':
                S = T
                continue
        k += 1
    return S


def decide_vector(types, items, vec, tl, workers, log, benders_frac=0.5, nogoods=None):
    """returns (status, placement or None, how). status in FEASIBLE / INFEASIBLE / UNKNOWN.
    placement: list of (item_index, bin_type_index_in_types, copy, x, y)."""
    t0 = time.time()
    deadline = t0 + tl
    if nogoods is None:
        nogoods = []
    bins = [(t, k) for t in range(len(types)) for k in range(vec[t])]
    # ---- (b) Benders loop
    bend_deadline = t0 + benders_frac * tl
    it = 0
    rel = None
    n_in_rel = 0
    while time.time() < bend_deadline:
        it += 1
        if rel is None:
            rel = VecModel(types, items, vec, geometry=False, nogoods=nogoods)
        else:  # incremental: add only the new no-goods to the existing master
            for S, (Wn, Hn) in nogoods[n_in_rel:]:
                rel.add_nogood(S, Wn, Hn)
        n_in_rel = len(nogoods)
        st = rel.solve(min(bend_deadline - time.time(), 0.5 * tl), workers)
        if st == 'INFEASIBLE':
            return 'INFEASIBLE', None, f'benders_it{it}_ng{len(nogoods)}'
        if st == 'UNKNOWN':
            break
        asg = rel.assignment()
        place = []
        stuck = False
        new = 0
        for b, S in enumerate(asg):
            t, k = bins[b]
            W, H = types[t][0], types[t][1]
            S = sorted(S)  # sorted by item index = non-increasing area
            sub = [items[i] for i in S]
            rem = bend_deadline - time.time()
            if rem <= 0:
                stuck = True
                break
            st2, xy = opp(sub, W, H, min(rem, max(1.0, 0.1 * tl)), workers)
            if st2 == 'FEASIBLE':
                place += [(S[q], t, k, xy[q][0], xy[q][1]) for q in range(len(S))]
            elif st2 == 'INFEASIBLE':
                S2 = shrink_nogood(items, S, W, H, min(1.0, 0.02 * tl), workers, bend_deadline)
                nogoods.append((tuple(S2), (W, H)))
                new += 1
            else:
                stuck = True
        if not stuck and new == 0:
            return 'FEASIBLE', place, f'benders_it{it}'
        if stuck and new == 0:
            break
    # ---- (c) full model
    rem = deadline - time.time()
    if rem <= 0.5:
        return 'UNKNOWN', None, f'benders_timeout_ng{len(nogoods)}'
    full = VecModel(types, items, vec, geometry=True, nogoods=nogoods)
    st = full.solve(rem, workers)
    if st == 'FEASIBLE':
        pl = [(i, bins[b][0], bins[b][1], x, y) for (i, b, x, y) in full.placement()]
        return 'FEASIBLE', pl, f'full_ng{len(nogoods)}'
    return st, None, f'full_ng{len(nogoods)}'


# --------------------------------------------------------------------------- solution output
def write_solution(nm, variant, types, items, placement, cost, tag=''):
    os.makedirs(SOL_DIR, exist_ok=True)
    place = []
    for i, t, k, x, y in placement:
        w, h = items[i]
        place.append([types[t][3], k, w, h, x, y])  # raw bin-line index
    fn = os.path.join(ROOT, ledger.sol_relpath(nm, variant, METHOD + tag, cost))
    json.dump(dict(name=nm, variant=variant, method=METHOD, cost=cost, place=place), open(fn, 'w'))
    out = subprocess.run([sys.executable, os.path.join(HERE, 'check_solution.py'), io.instance_path(nm),
                          fn, variant], capture_output=True, text=True)
    ok = out.stdout.startswith('valid') and out.stdout.strip().endswith(f'cost {cost}')
    return os.path.relpath(fn, ROOT), ok, out.stdout.strip() + out.stderr.strip()[-200:]


# --------------------------------------------------------------------------- driver
def run(nm, variant, tl_total, tl_vec, workers=2, ub=None, use_ledger_lb=False, cont=False,
        record=True, log=print, benders_frac=0.5, max_vectors=None, short=None, lb_ext=None,
        lb_ext_method=''):
    t_start = time.time()
    area = variant == 'area'
    bins_raw, items = io.read(nm, area)
    items = sorted(items, key=lambda it: (-it[0] * it[1], -it[0], -it[1]))
    types = reduce_types(bins_raw)
    best = ledger.best().get((nm, variant), {})
    ub_checked = best.get('ub') if best.get('ub_checked') else None
    cands = [v for v in (ub, ub_checked, io.bk[nm][0]) if v is not None]
    U = min(cands)
    U_is_checked = ub_checked is not None and U == ub_checked and ub is None
    L_ext = best.get('lb') if use_ledger_lb else None
    L_ext_method = best.get('lb_method', '') if L_ext is not None else ''
    if lb_ext is not None:  # explicit external LB (e.g. to replay a run without circularity)
        L_ext, L_ext_method = lb_ext, lb_ext_method
    vmax = U - 1 if U_is_checked else U
    vecs = enumerate_vectors(types, items, vmax)
    log(f'# {nm} {variant} n={len(items)} types={[(W, H, c) for W, H, c, _ in types]} '
        f'(raw {len(bins_raw)}) U={U} checked={U_is_checked} L_ext={L_ext} vectors(cost<={vmax})={len(vecs)}')
    stats = dict(enum=len(vecs), skipped_below_L=0, cheap=0, cp_inf=0, unknown=0)
    reasons = {}
    found = None          # (cost, vec, placement) of the cheapest vector proven feasible
    frontier = vmax + 1   # cost of the first vector NOT examined in pass 1 (all later ones untouched)
    unknown = []          # vectors examined but neither refuted nor packed (pass 1)
    nogoods = []
    processed = 0
    tl1 = short if short else tl_vec

    def decide(vec, cost, tl):
        t1 = time.time()
        st, pl, how = decide_vector(types, items, vec, tl, workers, log, benders_frac=benders_frac,
                                    nogoods=nogoods)
        log(f'  vec {vec} cost {cost}: {st} via {how} in {time.time() - t1:.1f}s (tl {tl:.0f})')
        return st, pl

    # ---- pass 1: cost order; short per-vector limit if `short`, else tl_vec
    for cost, vec in vecs:
        if time.time() - t_start > tl_total:
            log(f'# total time limit reached at cost {cost}')
            frontier = cost
            break
        if L_ext is not None and cost < L_ext:
            stats['skipped_below_L'] += 1
            continue
        r = cheap_filter(types, items, vec)
        if r is not None:
            stats['cheap'] += 1; reasons[r] = reasons.get(r, 0) + 1
            continue
        if max_vectors is not None and processed >= max_vectors:
            frontier = cost
            break
        processed += 1
        rem_total = tl_total - (time.time() - t_start)
        st, pl = decide(vec, cost, min(tl1, rem_total))
        if st == 'INFEASIBLE':
            stats['cp_inf'] += 1
            continue
        if st == 'FEASIBLE':
            found = (cost, vec, pl)
            frontier = cost
            break
        unknown.append((cost, vec))
        if not (cont or short):
            frontier = cost
            break
    # ---- pass 2 (only with `short`): revisit the undecided vectors in cost order with tl_vec;
    # stop at the first one that stays undecided (the LB cannot pass it).
    if short:
        for cost, vec in list(unknown):
            if found is not None and cost >= found[0]:
                break
            rem_total = tl_total - (time.time() - t_start)
            if rem_total < 1:
                break
            st, pl = decide(vec, cost, min(tl_vec, rem_total))
            if st == 'INFEASIBLE':
                stats['cp_inf'] += 1
                unknown.remove((cost, vec))
                continue
            if st == 'FEASIBLE':
                unknown.remove((cost, vec))
                found = (cost, vec, pl)
                frontier = min(frontier, cost)
            break
    stats['unknown'] = len(unknown)
    open_costs = [c for c, v in unknown if found is None or c < found[0]]
    lb = min([frontier] + open_costs)   # every examined/filtered vector cheaper than lb is refuted
    first_unknown = min(unknown) if unknown else None
    # the true LB is the smallest lattice value that is >= lb ... lb is a vector cost already
    # (or vmax+1 = U when U is a checked UB). If L_ext was used, LB relies on it.
    dt = time.time() - t_start
    res = dict(name=nm, variant=variant, lb=lb, ub=None, status='LB_ONLY', time=dt, stats=stats,
               reasons=reasons, first_unknown=first_unknown, ub_in=U, ub_in_checked=U_is_checked)
    sol_file, checked = '', 0
    if found is not None:
        cost, vec, pl = found
        sol_file, ok, msg = write_solution(nm, variant, types, items, pl, cost)
        checked = int(ok)
        log(f'  packing cost {cost} -> {sol_file}: {msg}')
        res['ub'] = cost
    ub_final = res['ub'] if res['ub'] is not None else (U if U_is_checked else None)
    if ub_final is not None and lb is not None and lb >= ub_final and (checked or res['ub'] is None):
        res['status'] = 'OPTIMAL'
        lb = ub_final; res['lb'] = lb
    elif res['ub'] is not None:
        res['status'] = 'FEASIBLE'
    note = (f"lattice enum={stats['enum']} cheap={stats['cheap']} cpinf={stats['cp_inf']} "
            f"unk={stats['unknown']} reasons={reasons} vec_tl={tl_vec}"
            + (f' LB relies on ledger LB {L_ext} from {L_ext_method}' if L_ext is not None else ' self-contained LB')
            + (f' first_unknown={first_unknown}' if first_unknown else ''))
    res['note'] = note
    if record:
        ledger.record(nm, variant, METHOD, res['status'], lb=lb, ub=res['ub'], time_s=dt, threads=workers,
                      sol_file=sol_file, checked=checked, note=note)
    log(f'RESULT {nm} {variant} status={res["status"]} lb={lb} ub={res["ub"]} U_in={U} '
        f'time={dt:.1f}s {note}')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variant', choices=['weighted', 'area'], required=True)
    ap.add_argument('--time', type=float, default=300.0, help='total time per instance (s)')
    ap.add_argument('--vec-time', type=float, default=60.0, help='time limit per vector (s)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--ub', type=int, default=None)
    ap.add_argument('--use-ledger-lb', action='store_true')
    ap.add_argument('--continue', dest='cont', action='store_true')
    ap.add_argument('--no-record', action='store_true')
    ap.add_argument('--benders-frac', type=float, default=0.5)
    ap.add_argument('--count-only', action='store_true')
    ap.add_argument('--short', type=float, default=None,
                    help='two-pass mode: pass 1 with this per-vector limit, pass 2 revisits undecided vectors')
    ap.add_argument('--method', default=METHOD, help='ledger method id (e.g. lattice_opp_fullonly for cross-checks)')
    ap.add_argument('names', nargs='+')
    a = ap.parse_args()
    METHOD = a.method
    for nm in a.names:
        if a.count_only:
            bins_raw, items = io.read(nm, a.variant == 'area')
            types = reduce_types(bins_raw)
            best = ledger.best().get((nm, a.variant), {})
            U = min(v for v in (a.ub, best.get('ub'), io.bk[nm][0]) if v is not None)
            vecs = enumerate_vectors(types, items, U)
            surv = [(c, v) for c, v in vecs if cheap_filter(types, sorted(items, key=lambda it: -it[0] * it[1]), v) is None]
            print(nm, a.variant, 'U', U, 'types', len(types), 'vectors', len(vecs), 'survive_cheap', len(surv),
                  'first', surv[:3], flush=True)
            continue
        run(nm, a.variant, a.time, a.vec_time, workers=a.workers, ub=a.ub, use_ledger_lb=a.use_ledger_lb,
            cont=a.cont, record=not a.no_record, benders_frac=a.benders_frac, short=a.short,
            log=lambda s: print(s, flush=True))
