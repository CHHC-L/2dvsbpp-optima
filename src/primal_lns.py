"""Primal track (sprint 2026-10-14): CP-SAT large-neighbourhood search on bins (single thread).

Start from the best CHECKED packing available for (instance, variant): any ledger row with
checked=1 and a sol_file, from either variant (a packing is feasible for both; it is re-costed).
Repeat until the time limit: pick 2-4 bins (biased to low cost-efficiency = item area / bin cost),
and re-pack their items EXACTLY-ish with a small CP-SAT 2DVSBPP model (per-type slots,
NoOverlap2D, 1 worker, short time limit) under the constraint cost <= (cost of the chosen bins) - 1.
Any solution found strictly improves the packing. The final packing is verified, written in
check_solution.py format, externally checked and recorded (method 'primal_lns').

usage: python src/primal_lns.py --variant area --tl 60 --sub-tl 2 --names MB_C3_21,...
"""
import argparse
import json
import os
import random
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger  # noqa: E402
import vsbpp_io as io  # noqa: E402
from primal_run import cost_of, verify  # noqa: E402
from primal_heur import Inst, store  # noqa: E402

ROOT = io.ROOT


def load_best_packing(nm, variant, bins, items):
    """Best checked packing over all ledger rows (both variants), re-costed for `variant`."""
    best = None
    seen = set()
    for r in ledger.rows():
        if r['name'] != nm or r['checked'] not in ('1',) or not r['sol_file']:
            continue
        f = os.path.join(ROOT, r['sol_file'])
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f)
        try:
            place = json.load(open(f))['place']
            place = [list(map(int, p)) for p in place]
            verify(place, bins, items)
        except Exception:
            continue
        c = cost_of(place, bins, variant)
        if best is None or c < best[0]:
            best = (c, place, r['sol_file'])
    return best


def to_bins(place):
    d = {}
    for t, k, w, h, x, y in place:
        d.setdefault((t, k), []).append((w, h, x, y))
    return [[t, v] for (t, k), v in d.items()]


def to_place(sol):
    place = []; kc = {}
    for t, v in sol:
        k = kc.get(t, 0); kc[t] = k + 1
        for w, h, x, y in v:
            place.append([t, k, w, h, x, y])
    return place


def subsolve(its, types, bins, cost, ub, tl, plateau=False):
    """its: [(w, h)], types: raw bin indices. Min-cost packing with cost <= ub. Returns [[t, [(w,h,x,y)]]] or None.
    plateau=True: secondary objective sum k * area_i * a[i,t,k] (push item area into low-index slots,
    i.e. concentrate the load so that some bin empties out), lexicographically after the cost."""
    n = len(its)
    order = sorted(range(n), key=lambda i: -its[i][0] * its[i][1])
    its = [its[i] for i in order]
    m = cp_model.CpModel()
    assign = [[] for _ in range(n)]
    slots = []
    obj = []
    sec = []
    for t in types:
        W, H = bins[t][:2]; c = cost[t]
        fit = [i for i in range(n) if its[i][0] <= W and its[i][1] <= H]
        if not fit:
            continue
        K = min(len(fit), ub // c)
        prev = None
        for k in range(K):
            u = m.NewBoolVar('')
            if prev is not None:
                m.AddImplication(u, prev)
            prev = u
            obj.append(c * u)
            xi = []; yi = []; ar = []; lits = []; rec = []
            for i in fit:
                if i < k:
                    continue
                w, h = its[i]
                a = m.NewBoolVar('')
                m.AddImplication(a, u)
                x = m.NewIntVar(0, W - w, ''); y = m.NewIntVar(0, H - h, '')
                xi.append(m.NewOptionalFixedSizeIntervalVar(x, w, a, ''))
                yi.append(m.NewOptionalFixedSizeIntervalVar(y, h, a, ''))
                ar.append(w * h); lits.append(a)
                assign[i].append(a)
                rec.append((i, a, x, y))
                if plateau and k > 0:
                    sec.append(k * w * h * a)
            if not lits:
                continue
            m.AddNoOverlap2D(xi, yi)
            m.Add(sum(ar[j] * lits[j] for j in range(len(lits))) <= W * H * u)
            m.AddCumulative(xi, [its[r[0]][1] for r in rec], H)
            m.AddCumulative(yi, [its[r[0]][0] for r in rec], W)
            slots.append((t, u, rec))
    for i in range(n):
        if not assign[i]:
            return None
        m.AddExactlyOne(assign[i])
    m.Add(sum(obj) <= ub)
    if plateau:
        big = sum(w * h for w, h in its) * (n + 1) + 1
        m.Minimize(big * sum(obj) + sum(sec))
    else:
        m.Minimize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.num_workers = 1
    s.parameters.max_time_in_seconds = tl
    try:
        st = s.Solve(m)
    except Exception as e:   # ortools 9.15 occasionally raises IndexError (raw_hash_map::at) inside Solve
        print('  CP-SAT exception ignored:', type(e).__name__, flush=True)
        return None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = []
    for t, u, rec in slots:
        if s.Value(u):
            v = [(its[i][0], its[i][1], s.Value(x), s.Value(y)) for i, a, x, y in rec if s.Value(a)]
            if v:
                out.append([t, v])
    return out


def lns(nm, variant, tl, sub_tl, seed=0, lb=None, maxit=30, log=print, stall=None, p_plateau=0.3,
        plateau_tl=0.4):
    random.seed(seed)
    I = Inst(nm, variant)
    bins, items, cost = I.bins, I.items, I.cost
    st = load_best_packing(nm, variant, bins, items)
    if st is None:
        return None
    c0, place, src = st
    sol = to_bins(place)
    cur = sum(cost[t] for t, _ in sol)
    t0 = time.time()
    nit = 0; nimp = 0; tlast = t0; npl = 0
    while time.time() - t0 < tl:
        if lb is not None and cur <= lb:
            break
        if stall is not None and time.time() - tlast > stall:
            break
        nb = len(sol)
        if nb < 2:
            break
        eff = [sum(w * h for w, h, _, _ in v) / cost[t] for t, v in sol]
        rank = sorted(range(nb), key=lambda b: eff[b])
        first = rank[min(int(random.expovariate(1.0 / max(1, nb / 4))), nb - 1)]
        k = random.choice([2, 2, 3, 3, 4])
        others = [b for b in range(nb) if b != first]
        random.shuffle(others)
        chosen = [first]
        nitems = len(sol[first][1])
        for b in others:
            if len(chosen) >= k:
                break
            if nitems + len(sol[b][1]) > maxit:
                continue
            chosen.append(b); nitems += len(sol[b][1])
        if len(chosen) < 2:
            nit += 1
            continue
        its = [(w, h) for b in chosen for w, h, _, _ in sol[b][1]]
        cs = sum(cost[sol[b][0]] for b in chosen)
        plateau = random.random() < p_plateau
        if plateau:
            res = subsolve(its, I.types, bins, cost, cs, min(plateau_tl, sub_tl), plateau=True)
        else:
            res = subsolve(its, I.types, bins, cost, cs - 1, sub_tl)
        nit += 1
        if res is not None and plateau and sum(cost[t] for t, _ in res) == cs:
            assert sorted(its) == sorted((w, h) for _, v in res for w, h, _, _ in v)
            sol = [sol[b] for b in range(nb) if b not in chosen] + res
            npl += 1
            continue
        if res is not None:
            newc = sum(cost[t] for t, _ in res)
            assert newc < cs
            assert sorted(its) == sorted((w, h) for _, v in res for w, h, _, _ in v)
            sol = [sol[b] for b in range(nb) if b not in chosen] + res
            cur = sum(cost[t] for t, _ in sol)
            nimp += 1; tlast = time.time()
            log(f'  {nm} {variant} it={nit} {cs}->{newc} total {cur} t={time.time() - t0:.1f}')
    log(f'  {nm} {variant} plateau moves accepted {npl}')
    return to_place(sol), c0, cur, nit, nimp, src, I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='area', choices=['weighted', 'area'],
                    help='default variant; a names entry NAME:variant overrides it')
    ap.add_argument('--tl', type=float, default=60)
    ap.add_argument('--sub-tl', type=float, default=2.0)
    ap.add_argument('--names', required=True)
    ap.add_argument('--tag', default='l1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--p-plateau', type=float, default=0.3)
    ap.add_argument('--stall', type=float, default=None, help='stop an instance after this many s without improvement')
    ap.add_argument('--stop-at', default='', help='HH:MM local wall clock: do not start new instances after')
    a = ap.parse_args()
    default_variant = a.variant
    for entry in a.names.split(','):
        nm, var = (entry.split(':') + [default_variant])[:2]
        a.variant = var
        other = 'area' if a.variant == 'weighted' else 'weighted'
        if a.stop_at and time.strftime('%H:%M') >= a.stop_at and time.strftime('%H') != '23':
            print('stop-at reached', flush=True)
            break
        bd = ledger.best()
        lb = bd.get((nm, a.variant), {}).get('lb')
        t0 = time.time()
        r = lns(nm, a.variant, a.tl, a.sub_tl, a.seed, lb=lb, log=lambda s: print(s, flush=True), stall=a.stall,
                p_plateau=a.p_plateau)
        if r is None:
            print(nm, a.variant, 'no start packing', flush=True)
            continue
        place, c0, cur, nit, nimp, src, I = r
        t_s = time.time() - t0
        verify(place, I.bins, I.items)
        c = cost_of(place, I.bins, a.variant)
        assert c == cur
        msg = [nm, a.variant, 'start', c0, 'final', c, 'its', nit, 'imp', nimp, round(t_s, 1), 'lb', lb,
               'csv', io.bk[nm]]
        bv = ledger.best().get((nm, a.variant), {}).get('ub')
        # store if LNS improved, or if the start packing (possibly re-costed from the other variant)
        # already beats this variant's best checked UB and so needs its own file
        if c < c0 or bv is None or c < bv:
            c, ok = store(nm, a.variant, place, I.bins, I.items, 'primal_lns', a.tag, a.tl, t_s,
                          f'start {c0} from {os.path.basename(src)}; sub_tl={a.sub_tl} its={nit} imp={nimp}')
            msg += ['rec', ok]
            b2 = ledger.best().get((nm, other), {})
            c2 = cost_of(place, I.bins, other)
            if b2.get('ub') is None or c2 < b2['ub']:
                c2, ok2 = store(nm, other, place, I.bins, I.items, 'primal_lns', a.tag + 'x', a.tl, t_s,
                                f'packing of {a.variant} lns re-costed')
                msg += ['cross', other, c2, ok2]
        else:  # no new packing: record the run without a UB (the start packing is already on file)
            ledger.record(nm, a.variant, 'primal_lns', 'NO_IMPROVEMENT', time_s=t_s, threads=1,
                          note=f'tag={a.tag} tl={a.tl}; no improvement over start {c0} from {os.path.basename(src)}')
        print(*msg, flush=True)


if __name__ == '__main__':
    main()
