"""Primal track (sprint 2026-10-14): own randomized MaxRects heuristic for the 2DVSBPP (oriented,
non-guillotine, unlimited bins per type), single thread, pure Python. Fallback / complement to
packingsolver (which in 1-thread mode gives no solution within seconds at n = 80-100).

Construction (bin by bin): for every bin type t, fill one empty bin of type t with the remaining
items (MaxRects, item order + placement rule given), score = packed item area / c_t (cost per
area efficiency; c_t = file cost (weighted) or W*H (area)), pick the best type (with random noise),
commit, repeat. Improvement: (1) downgrade every bin to the cheapest type that still holds its items,
(2) merge two bins into one cheaper bin, (3) try to empty the least efficient bin by moving its items
into other bins (each move re-packs the target bin from scratch). Randomized restarts until the time
limit; the best packing is verified, written in check_solution.py format, externally checked, and
recorded in the ledger (method 'primal_heur'), plus re-costed for the other variant when it helps.

usage: python src/primal_heur.py --variant weighted --tl 2 --tag h1 [--names ...] [--seed 0]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger  # noqa: E402
import vsbpp_io as io  # noqa: E402
from primal_run import read_raw, cost_of, verify, check_external  # noqa: E402

ROOT = io.ROOT


def mr_pack(W, H, seq, items, rule=0):
    """MaxRects. rule 0 = best short side fit, 1 = bottom-left, 2 = best area fit.
    Returns (placed [(i, x, y)], left [i])."""
    free = [(0, 0, W, H)]
    placed = []; left = []
    for i in seq:
        w, h = items[i]
        best = None
        for (fx, fy, fw, fh) in free:
            if w <= fw and h <= fh:
                if rule == 0:
                    a, b = fw - w, fh - h
                    s = (a, b) if a < b else (b, a)
                elif rule == 1:
                    s = (fy + h, fx)
                else:
                    s = (fw * fh - w * h, min(fw - w, fh - h))
                if best is None or s < best[0]:
                    best = (s, fx, fy)
        if best is None:
            left.append(i)
            continue
        _, x, y = best
        placed.append((i, x, y))
        nf = []
        xw, yh = x + w, y + h
        for r in free:
            fx, fy, fw, fh = r
            if x >= fx + fw or xw <= fx or y >= fy + fh or yh <= fy:
                nf.append(r)
                continue
            if x > fx:
                nf.append((fx, fy, x - fx, fh))
            if xw < fx + fw:
                nf.append((xw, fy, fx + fw - xw, fh))
            if y > fy:
                nf.append((fx, fy, fw, y - fy))
            if yh < fy + fh:
                nf.append((fx, yh, fw, fy + fh - yh))
        nf.sort(key=lambda r: -r[2] * r[3])
        pr = []
        for r in nf:
            rx, ry, rw, rh = r
            for q in pr:
                if rx >= q[0] and ry >= q[1] and rx + rw <= q[0] + q[2] and ry + rh <= q[1] + q[3]:
                    break
            else:
                pr.append(r)
        free = pr
    return placed, left


ORDER_KEYS = [
    lambda it: (-it[1], -it[0]),            # height desc
    lambda it: (-it[0], -it[1]),            # width desc
    lambda it: (-it[0] * it[1], -it[1]),    # area desc
    lambda it: (-max(it), -min(it)),        # max side desc
    lambda it: (-(it[0] + it[1]), -it[1]),  # perimeter desc
]


class Inst:
    def __init__(self, nm, variant):
        self.nm = nm
        self.bins, self.items = read_raw(nm)
        self.variant = variant
        self.cost = [(W * H if variant == 'area' else c) for (W, H, c) in self.bins]
        self.n = len(self.items)
        self.area = [w * h for (w, h) in self.items]
        # distinct (W, H, cost) types, keep raw index of the first occurrence
        seen = {}
        for t, (W, H, c) in enumerate(self.bins):
            key = (W, H, self.cost[t])
            if key not in seen:
                seen[key] = t
        # drop dominated types (another type at least as large in both dims and not more costly)
        T = list(seen.values())
        keep = []
        for t in T:
            W, H = self.bins[t][:2]
            dom = False
            for u in T:
                if u == t:
                    continue
                W2, H2 = self.bins[u][:2]
                if W2 >= W and H2 >= H and self.cost[u] <= self.cost[t] and \
                        (W2, H2, self.cost[u]) != (W, H, self.cost[t]):
                    dom = True
                    break
            if not dom:
                keep.append(t)
        self.types = sorted(keep, key=lambda t: self.cost[t])
        self.orders = []
        for kf in ORDER_KEYS:
            self.orders.append(sorted(range(self.n), key=lambda i: kf(self.items[i])))

    def fits_dims(self, t, idxs):
        W, H = self.bins[t][:2]
        return all(self.items[i][0] <= W and self.items[i][1] <= H for i in idxs)

    def pack_set(self, t, idxs, tries=None):
        """Try to pack ALL items idxs into one bin of type t. Returns placement or None."""
        W, H = self.bins[t][:2]
        if sum(self.area[i] for i in idxs) > W * H or not self.fits_dims(t, idxs):
            return None
        s = set(idxs)
        for rule in (0, 1):
            for od in self.orders:
                seq = [i for i in od if i in s]
                pl, left = mr_pack(W, H, seq, self.items, rule)
                if not left:
                    return pl
        if tries:
            lst = list(idxs)
            for _ in range(tries):
                random.shuffle(lst)
                lst.sort(key=lambda i: -self.area[i] * random.uniform(0.6, 1.4))
                pl, left = mr_pack(W, H, lst, self.items, random.randint(0, 2))
                if not left:
                    return pl
        return None


def construct(I, order, rule, noise):
    remaining = list(order)
    sol = []   # list of [t, placement]
    while remaining:
        cands = []
        rem_area = sum(I.area[i] for i in remaining)
        for t in I.types:
            W, H = I.bins[t][:2]
            pl, left = mr_pack(W, H, remaining, I.items, rule)
            if not pl:
                continue
            a = sum(I.area[i] for i, _, _ in pl)
            eff = a / I.cost[t]
            if not left:
                # this bin finishes the instance; prefer the cheapest such bin
                eff = rem_area / I.cost[t] * 1.0001
            cands.append((eff * (1 + noise * random.random()), t, pl, left))
        cands.sort(key=lambda c: -c[0])
        _, t, pl, left = cands[0]
        sol.append([t, pl])
        remaining = left
    return sol


def sol_cost(I, sol):
    return sum(I.cost[t] for t, _ in sol)


def improve(I, sol, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # (1) downgrade
        for b in sol:
            t, pl = b
            idxs = [i for i, _, _ in pl]
            for u in I.types:
                if I.cost[u] >= I.cost[t]:
                    break
                p2 = I.pack_set(u, idxs)
                if p2 is not None:
                    b[0], b[1] = u, p2
                    improved = True
                    break
        # (2) merge pairs
        sol.sort(key=lambda b: sum(I.area[i] for i, _, _ in b[1]) / I.cost[b[0]])
        done = False
        for a in range(len(sol)):
            if time.time() > deadline:
                break
            for b in range(a + 1, len(sol)):
                ia = [i for i, _, _ in sol[a][1]]; ib = [i for i, _, _ in sol[b][1]]
                cab = I.cost[sol[a][0]] + I.cost[sol[b][0]]
                idxs = ia + ib
                tot = sum(I.area[i] for i in idxs)
                for u in I.types:
                    if I.cost[u] >= cab:
                        break
                    W, H = I.bins[u][:2]
                    if W * H < tot:
                        continue
                    p2 = I.pack_set(u, idxs)
                    if p2 is not None:
                        sol[a] = [u, p2]
                        del sol[b]
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if done:
            improved = True
            continue
        # (3) empty the least efficient bin(s)
        sol.sort(key=lambda b: sum(I.area[i] for i, _, _ in b[1]) / I.cost[b[0]])
        for a in range(min(3, len(sol))):
            if time.time() > deadline:
                break
            src = sol[a]
            work = {j: [i for i, _, _ in sol[j][1]] for j in range(len(sol)) if j != a}
            newpl = {}
            ok = True
            for i, _, _ in sorted(src[1], key=lambda p: -I.area[p[0]]):
                placed = False
                for j in sorted(work, key=lambda j: -(I.bins[sol[j][0]][0] * I.bins[sol[j][0]][1]
                                                       - sum(I.area[q] for q in work[j]))):
                    t = sol[j][0]
                    p2 = I.pack_set(t, work[j] + [i])
                    if p2 is not None:
                        work[j].append(i); newpl[j] = p2; placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                for j, p2 in newpl.items():
                    sol[j][1] = p2
                del sol[a]
                improved = True
                break
    return sol


def solve(nm, variant, tl, seed=0, ub_known=None):
    random.seed(seed)
    I = Inst(nm, variant)
    t0 = time.time(); deadline = t0 + tl
    best = None; bestc = None; it = 0
    while True:
        k = it % (len(I.orders) * 3)
        rule = k // len(I.orders)
        if it < len(I.orders) * 3:
            order = I.orders[k % len(I.orders)]; noise = 0.0
        else:
            base = I.orders[random.randrange(len(I.orders))]
            order = sorted(base, key=lambda i: base.index(i) + random.gauss(0, 3))
            noise = random.choice([0.0, 0.05, 0.15, 0.3])
            rule = random.randint(0, 2)
        sol = construct(I, order, rule, noise)
        sol = improve(I, sol, deadline)
        c = sol_cost(I, sol)
        if bestc is None or c < bestc:
            best, bestc = [[t, list(pl)] for t, pl in sol], c
        it += 1
        if time.time() > deadline:
            break
    place = []
    kc = {}
    for t, pl in best:
        k = kc.get(t, 0); kc[t] = k + 1
        for i, x, y in pl:
            w, h = I.items[i]
            place.append([t, k, w, h, x, y])
    return place, I, it


def store(nm, variant, place, bins, items, method, tag, tl, t_s, note):
    cost = cost_of(place, bins, variant)
    solrel = ledger.sol_relpath(nm, variant, method, cost, tag)
    solf = os.path.join(ROOT, solrel)
    tmp = solf + '.tmp'
    json.dump({'name': nm, 'variant': variant, 'cost': cost, 'method': method, 'tag': tag,
               'time_limit': tl, 'threads': 1, 'note': note, 'place': place}, open(tmp, 'w'))
    c_ext = check_external(nm, tmp, variant)
    ok = (c_ext == cost)
    if ok:
        os.replace(tmp, solf)
    else:
        os.remove(tmp)
    ledger.record(nm, variant, method, 'FEASIBLE', ub=cost, time_s=t_s, threads=1,
                  sol_file=solrel if ok else '', checked=1 if ok else 0,
                  note=f'tag={tag} tl={tl}; {note}' + ('' if ok else f'; CHECK FAILED ext={c_ext}'))
    return cost, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['weighted', 'area'])
    ap.add_argument('--tl', type=float, default=2.0)
    ap.add_argument('--tag', default='h1')
    ap.add_argument('--names', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only-if-better', type=int, default=0,
                    help='record only if better than the current ledger best UB of that variant')
    ap.add_argument('--dry', type=int, default=0)
    a = ap.parse_args()
    names = a.names.split(',') if a.names else io.all_names()
    done = set()
    for r in ledger.rows():
        if r['method'] == 'primal_heur' and f'tag={a.tag} ' in r['note'] + ' ' and r['variant'] == a.variant:
            done.add(r['name'])
        if r['method'] == 'primal_heur' and r['note'].startswith(f'tag={a.tag};') and r['variant'] == a.variant:
            done.add(r['name'])
    other = 'area' if a.variant == 'weighted' else 'weighted'
    for nm in names:
        if nm in done:
            continue
        t0 = time.time()
        place, I, iters = solve(nm, a.variant, a.tl, a.seed)
        t_s = time.time() - t0
        verify(place, I.bins, I.items)
        c = cost_of(place, I.bins, a.variant)
        msg = [nm, a.variant, c, 'iters', iters, round(t_s, 1)]
        if a.dry:
            print(*msg, 'bks', io.bk[nm], flush=True)
            continue
        bestd = ledger.best()
        b = bestd.get((nm, a.variant), {})
        note = f'seed={a.seed} iters={iters}'
        if not a.only_if_better or b.get('ub') is None or c < b['ub']:
            c, ok = store(nm, a.variant, place, I.bins, I.items, 'primal_heur', a.tag, a.tl, t_s, note)
            msg += ['rec', ok]
        b2 = bestd.get((nm, other), {})
        c2 = cost_of(place, I.bins, other)
        if b2.get('ub') is None or c2 < b2['ub']:
            c2, ok2 = store(nm, other, place, I.bins, I.items, 'primal_heur', a.tag + 'x', a.tl, t_s,
                            f'packing of {a.variant} run re-costed; ' + note)
            msg += ['cross', other, c2, ok2]
        print(*msg, flush=True)


if __name__ == '__main__':
    main()
