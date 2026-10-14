"""opp track: shared helpers for 2D orthogonal-packing (2D-OPP) feasibility proofs.

  * load(name)            raw bins [(W, H, file_cost)] (raw line order kept!) and items [(w, h)]
  * normal_patterns(...)  Herz / Christofides-Whitlock normal positions (subset sums of the
                          OTHER items' sizes, capped)
  * dff_check(items,W,H)  dual-feasible-function bounds (Fekete-Schepers u^k, CCM f0 / BM f2,
                          identity) in every pairing f(w) x g(h); returns a witness if infeasible
  * bin_vectors(...)      every bin-count vector with cost in [lo, hi] (distinct (W,H) types)
  * verify_place(...)     geometry check of a placement

Validity of the DFF test: if f, g are (integer) dual-feasible functions on [0,W] and [0,H]
(sum of f(w_i) over items fitting side-by-side in W is <= f(W)), then for any feasible 2D packing
sum_i f(w_i) g(h_i) <= f(W) g(H) (Fekete & Schepers 2004). All arithmetic here is exact integer.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    L = open(os.path.join(ROOT, 'data', 'instances', name + '.txt')).read().split('\n')
    n, m, _ = map(int, L[0].split())
    bins = [tuple(map(int, L[1 + t].split())) for t in range(m)]
    items = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
    return [(b[0], b[1], b[3]) for b in bins], items


def cost_of(bin_, variant):
    W, H, c = bin_
    return W * H if variant == 'area' else c


# ---------------------------------------------------------------------------------------------
# normal patterns
def subset_sums(sizes, cap):
    reach = 1  # bitset
    mask = (1 << (cap + 1)) - 1
    for s in sizes:
        reach = (reach | (reach << s)) & mask
    return [v for v in range(cap + 1) if (reach >> v) & 1]


def normal_patterns(sizes, i, cap):
    """positions for item i: subset sums of sizes of the other items, <= cap (= W - w_i)."""
    return subset_sums([s for j, s in enumerate(sizes) if j != i], cap)


# ---------------------------------------------------------------------------------------------
# dual-feasible functions (integer valued).  Each returns (name, table[0..C], f(C)).
def dff_identity(C):
    return ('id', list(range(C + 1)), C)


def dff_fs(C, k):
    """Fekete-Schepers u^(k), scaled by k*C: f(a) = k*a if C | (k+1)a else C*floor((k+1)a/C)."""
    t = []
    for a in range(C + 1):
        if ((k + 1) * a) % C == 0:
            t.append(k * a)
        else:
            t.append(C * (((k + 1) * a) // C))
    return (f'fs{k}', t, k * C)


def dff_ccm0(C, lam):
    """CCM f0^lam, 1 <= lam <= C/2: 0 below lam, C above C-lam, identity in between."""
    t = [0 if a < lam else (C if a > C - lam else a) for a in range(C + 1)]
    return (f'ccm0_{lam}', t, C)


def dff_bm2(C, lam):
    """Carlier-Clautiaux-Moukrim f2^lam (Boschetti-Mingozzi type), 1 <= lam <= C/2."""
    t = []
    for a in range(C + 1):
        if 2 * a > C:
            t.append(2 * (C // lam - (C - a) // lam))
        elif 2 * a == C:
            t.append(C // lam)
        else:
            t.append(2 * (a // lam))
    return (f'bm2_{lam}', t, 2 * (C // lam))


def dff_family(C, kmax=None):
    fs = [dff_identity(C)]
    for k in range(1, (kmax or C) + 1):
        fs.append(dff_fs(C, k))
    for lam in range(1, C // 2 + 1):
        fs.append(dff_ccm0(C, lam))
        fs.append(dff_bm2(C, lam))
    # sanity: verify DFF property for every function on all pairs/triples of sizes?  the
    # superadditivity-type condition is checked exhaustively by _check_dff in tests.
    return fs


def _check_dff(f, C):
    """Exhaustive DFF check: for every multiset of sizes with sum <= C, sum f <= f(C).
    Equivalent to: max over 1D knapsack of f-values with capacity C equals f(C) (unbounded)."""
    name, t, fc = f
    best = [0] * (C + 1)
    for cap in range(1, C + 1):
        b = best[cap - 1]
        for a in range(1, cap + 1):
            v = t[a] + best[cap - a]
            if v > b:
                b = v
        best[cap] = b
    return best[C] <= fc and t[0] == 0


def dff_check(items, W, H, check_family=True):
    """Return None if no DFF pair proves infeasibility, else witness dict."""
    if any(w > W or h > H for w, h in items):
        return dict(kind='too_big')
    if sum(w * h for w, h in items) > W * H:
        return dict(kind='area', lhs=sum(w * h for w, h in items), rhs=W * H)
    FX = [f for f in dff_family(W)]
    FY = [g for g in dff_family(H)]
    if check_family:
        FX = [f for f in FX if _check_dff(f, W)]
        FY = [g for g in FY if _check_dff(g, H)]
    best = None
    for fn, ft, fc in FX:
        fw = [ft[w] for w, h in items]
        for gn, gt, gc in FY:
            lhs = sum(fw[i] * gt[h] for i, (w, h) in enumerate(items))
            rhs = fc * gc
            if lhs > rhs:
                return dict(kind='dff', f=fn, g=gn, lhs=lhs, rhs=rhs)
            r = lhs / rhs if rhs else 0
            if best is None or r > best[0]:
                best = (r, fn, gn)
    return None if best is None else None


def dff_best_ratio(items, W, H):
    FX = [f for f in dff_family(W) if _check_dff(f, W)]
    FY = [g for g in dff_family(H) if _check_dff(g, H)]
    best = (0, None, None)
    for fn, ft, fc in FX:
        for gn, gt, gc in FY:
            lhs = sum(ft[w] * gt[h] for w, h in items)
            r = lhs / (fc * gc)
            if r > best[0]:
                best = (r, fn, gn)
    return best


# ---------------------------------------------------------------------------------------------
def bin_vectors(bins, variant, lo, hi):
    """All count vectors over DISTINCT (W,H) types (first raw index kept) with lo <= cost <= hi.
    returns list of (cost, [(t_raw, count), ...])."""
    types = []
    seen = set()
    for t, b in enumerate(bins):
        if (b[0], b[1]) in seen:
            # same geometry listed twice: keep the cheaper cost (unlimited supply)
            for q, (tt, bb) in enumerate(types):
                if (bb[0], bb[1]) == (b[0], b[1]) and cost_of(b, variant) < cost_of(bb, variant):
                    types[q] = (t, b)
            continue
        seen.add((b[0], b[1]))
        types.append((t, b))
    out = []

    def rec(q, cost, vec):
        if cost > hi:
            return
        if q == len(types):
            if cost >= lo:
                out.append((cost, [(t, c) for (t, c) in vec if c > 0]))
            return
        t, b = types[q]
        c = cost_of(b, variant)
        k = 0
        while cost + k * c <= hi:
            rec(q + 1, cost + k * c, vec + [(t, k)])
            k += 1
    rec(0, 0, [])
    out.sort()
    return out


def verify_place(items, bins_used, place):
    """items: list (w,h); bins_used: dict key -> (W,H); place: list (key, w, h, x, y)."""
    if sorted((p[1], p[2]) for p in place) != sorted(items):
        return False
    per = {}
    for key, w, h, x, y in place:
        W, H = bins_used[key]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False
        per.setdefault(key, []).append((x, y, w, h))
    for rs in per.values():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                xa, ya, wa, ha = rs[a]; xb, yb, wb, hb = rs[b]
                if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                    return False
    return True
