"""opp track: logic-based Benders for the 2DVSBPP with the OPP prover as subproblem.

Master (CP-SAT, a RELAXATION of the 2DVSBPP): per-type bin slots (as in cpsat_direct.py), items
assigned to slots, per-slot area constraint, lazily added DFF capacity rows and no-good cuts;
objective min sum of costs, with cost <= UB - 1 (we only look for something cheaper than UB).
Subproblem: for every used slot, decide 2D-OPP of its item set:
    1. DFF test (opp_core.dff_check; valid by Fekete-Schepers) -> add the violated DFF row to
       all slots of that bin type;
    2. CP-SAT OPP (opp_cpsat.opp_single) -> if INFEASIBLE: shrink the set to a minimal
       infeasible subset S (greedy deletion, each step re-proved), add the no-good
       sum_{i in S} a[i,t',k] <= |S|-1 for every slot of every type t' with W'<=W and H'<=H.
       UNKNOWN (time-out) -> nothing is cut; the run cannot finish with a proof.
    If every used slot is feasible -> a checked packing of cost = master optimum = proven optimum.
Validity of the final bound: every cut removes only geometrically infeasible bin contents; the
master's proven optimum (or its best bound at a time-out) is therefore a valid lower bound.
If the master becomes INFEASIBLE under cost <= UB - 1 then LB = UB.

usage: python src/opp_benders.py --variant area --time 300 MB_C8_1 [...]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402
import opp_cpsat as ocp  # noqa: E402


def _ub_default(name, variant):
    import vsbpp_io as io
    ub = io.bk[name][0]
    try:
        import ledger
        b = ledger.best().get((name, variant))
        if b and b['ub'] is not None and b['ub_checked'] and b['ub'] < ub:
            ub = b['ub']
    except Exception:
        pass
    return ub


class Benders:
    def __init__(self, name, variant, ub, workers=1, opp_tl=30.0, log=print):
        self.name, self.variant, self.ub = name, variant, ub
        raw_bins, items = oc.load(name)
        self.raw_bins = raw_bins
        # distinct geometries, cheapest raw line kept
        types = {}
        for t, b in enumerate(raw_bins):
            c = oc.cost_of(b, variant)
            key = (b[0], b[1])
            if key not in types or c < types[key][2]:
                types[key] = (t, key, c)
        self.types = sorted(types.values(), key=lambda r: r[0])  # (t_raw, (W,H), cost)
        order = sorted(range(len(items)), key=lambda i: (-items[i][0] * items[i][1], i))
        self.items = [items[i] for i in order]
        self.workers, self.opp_tl, self.log = workers, opp_tl, log
        self.dff_rows = {}   # type q -> list of (ftab, gtab, cap, label)
        self.nogoods = []    # (W, H, frozenset of item indices)
        self.cache = {}      # (W,H,frozenset) -> status
        self.n_opp = 0
        self.dff_fam = {}
        # initial DFF rows: half-height / half-width large-item rows (CCM f0 with lam = C//2)
        # (validated by the exhaustive DFF check before use)
        for q, (traw, (W, H), c) in enumerate(self.types):
            for ft, gt, lab in ((list(range(W + 1)), oc.dff_ccm0(H, max(1, H // 2))[1], 'idxccm0'),
                                (oc.dff_ccm0(W, max(1, W // 2))[1], list(range(H + 1)), 'ccm0xid')):
                if oc._check_dff(('f', ft, ft[W]), W) and oc._check_dff(('g', gt, gt[H]), H):
                    self.dff_rows.setdefault(q, []).append((ft, gt, ft[W] * gt[H], lab))

    def _family(self, C):
        if C not in self.dff_fam:
            self.dff_fam[C] = [f for f in oc.dff_family(C) if oc._check_dff(f, C)]
        return self.dff_fam[C]

    def _dff_violation(self, S, W, H):
        its = [self.items[i] for i in S]
        best = None
        for fn, ft, fc in self._family(W):
            fw = [ft[w] for w, h in its]
            for gn, gt, gc in self._family(H):
                lhs = sum(fw[j] * gt[h] for j, (w, h) in enumerate(its))
                if lhs > fc * gc:
                    r = lhs / (fc * gc)
                    if best is None or r > best[0]:
                        best = (r, ft, gt, fc * gc, f'{fn}x{gn}')
        return best

    def master(self, tl):
        from ortools.sat.python import cp_model
        items, n = self.items, len(self.items)
        m = cp_model.CpModel()
        a = {}; u = {}; cost_terms = []
        for q, (traw, (W, H), c) in enumerate(self.types):
            fit = [i for i in range(n) if items[i][0] <= W and items[i][1] <= H]
            if not fit:
                continue
            K = min(len(fit), (self.ub - 1) // c)
            for k in range(K):
                u[q, k] = m.NewBoolVar('')
                cost_terms.append(c * u[q, k])
                lits = []
                for i in fit:
                    if i < k:
                        continue
                    a[i, q, k] = m.NewBoolVar('')
                    m.AddImplication(a[i, q, k], u[q, k])
                    lits.append(i)
                m.Add(sum(items[i][0] * items[i][1] * a[i, q, k] for i in lits) <= W * H * u[q, k])
                m.AddBoolOr([a[i, q, k] for i in lits] + [u[q, k].Not()])
                for ft, gt, cap, lab in self.dff_rows.get(q, []):
                    m.Add(sum(ft[items[i][0]] * gt[items[i][1]] * a[i, q, k] for i in lits) <= cap)
                # pairwise conflicts: neither side by side nor stacked -> not in the same bin
                for x1 in range(len(lits)):
                    i = lits[x1]
                    for x2 in range(x1 + 1, len(lits)):
                        j = lits[x2]
                        if items[i][0] + items[j][0] > W and items[i][1] + items[j][1] > H:
                            m.AddBoolOr([a[i, q, k].Not(), a[j, q, k].Not()])
                for (Wn, Hn, S) in self.nogoods:
                    if W <= Wn and H <= Hn and all((i, q, k) in a for i in S):
                        m.Add(sum(a[i, q, k] for i in S) <= len(S) - 1)
            for k in range(K - 1):
                m.AddImplication(u[q, k + 1], u[q, k])
        for i in range(n):
            m.AddExactlyOne([v for (j, q, k), v in a.items() if j == i])
        obj = sum(cost_terms)
        m.Add(obj <= self.ub - 1)
        m.Minimize(obj)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = tl
        s.parameters.num_workers = self.workers
        st = s.Solve(m)
        if st == cp_model.INFEASIBLE:
            return 'INFEASIBLE', None, self.ub
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            lb = int(math.ceil(s.BestObjectiveBound() - 1e-6))
            bins = {}
            for (i, q, k), v in a.items():
                if s.Value(v):
                    bins.setdefault((q, k), []).append(i)
            return s.StatusName(st), (int(round(s.ObjectiveValue())), bins), lb
        lb = int(math.ceil(s.BestObjectiveBound() - 1e-6)) if st != cp_model.MODEL_INVALID else None
        return 'UNKNOWN', None, lb

    def opp(self, S, W, H):
        key = (W, H, frozenset(S))
        if key in self.cache:
            return self.cache[key]
        self.n_opp += 1
        st, pl, dt = ocp.opp_single([self.items[i] for i in S], W, H, tl=self.opp_tl, workers=1)
        self.cache[key] = (st, pl)
        return st, pl

    def minimise(self, S, W, H):
        S = list(S)
        for i in sorted(S, key=lambda i: self.items[i][0] * self.items[i][1]):
            T = [j for j in S if j != i]
            if not T:
                continue
            if self._dff_violation(T, W, H) is not None or sum(self.items[j][0] * self.items[j][1] for j in T) > W * H:
                S = T; continue
            st, _ = self.opp(T, W, H)
            if st == 'INFEASIBLE':
                S = T
        return S

    def run(self, total_tl=300.0, master_tl=120.0):
        t0 = time.time(); it = 0; best_lb = None; result = None
        while time.time() - t0 < total_tl:
            it += 1
            st, sol, lb = self.master(min(master_tl, max(5.0, total_tl - (time.time() - t0))))
            if lb is not None:  # CP-SAT bound of the relaxation master: valid at any status
                lb = min(lb, self.ub)  # master has cost <= ub-1; a bound above that only means 'nothing <= ub-1'
                best_lb = lb if best_lb is None else max(best_lb, lb)
            if st == 'INFEASIBLE':
                result = dict(status='OPTIMAL_UB', lb=self.ub, ub=None)
                self.log(f'it {it}: master infeasible with cost <= {self.ub - 1} -> LB = {self.ub}')
                break
            if st == 'UNKNOWN' or st == 'FEASIBLE':
                # FEASIBLE (not optimal) master: we may still check its bins, but the bound is lb
                if st == 'UNKNOWN':
                    self.log(f'it {it}: master unknown, bound {lb}')
                    result = dict(status='UNKNOWN', lb=best_lb, ub=None)
                    break
            val, bins = sol
            cut = 0; allok = True; placement = []; unknown = False
            for (q, k), S in bins.items():
                traw, (W, H), c = self.types[q]
                v = self._dff_violation(S, W, H)
                if v is not None:
                    self.dff_rows.setdefault(q, []).append((v[1], v[2], v[3], v[4]))
                    cut += 1; allok = False
                    continue
                stq, pl = self.opp(S, W, H)
                if stq == 'FEASIBLE':
                    placement += [[traw, k, w, h, x, y] for (w, h, x, y) in pl]
                    continue
                allok = False
                if stq == 'INFEASIBLE':
                    Smin = self.minimise(S, W, H)
                    self.nogoods.append((W, H, frozenset(Smin)))
                    cut += 1
                else:
                    self.log(f'it {it}: OPP UNKNOWN for bin {W}x{H} set {S}')
                    unknown = True
            self.log(f'it {it}: master {st} val {val} lb {lb} bins {len(bins)} cuts {cut} '
                     f'(dff rows {sum(len(v) for v in self.dff_rows.values())}, nogoods {len(self.nogoods)}, '
                     f'opp calls {self.n_opp}) t={time.time() - t0:.1f}')
            if unknown and cut == 0:
                # cannot cut the master solution: stop; the master optimum is still a valid LB
                result = dict(status='UNKNOWN', lb=best_lb, ub=None)
                break
            if allok:
                result = dict(status='OPTIMAL' if st == 'OPTIMAL' else 'FEASIBLE', lb=lb if st == 'OPTIMAL' else lb,
                              ub=val, place=placement)
                break
        if result is None:
            result = dict(status='TIMEOUT', lb=best_lb, ub=None)
        result.update(iters=it, time=time.time() - t0, nogoods=[(W, H, sorted(S)) for W, H, S in self.nogoods],
                      dff_rows=sum(len(v) for v in self.dff_rows.values()), opp_calls=self.n_opp)
        return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variant', choices=['weighted', 'area'], required=True)
    ap.add_argument('--time', type=float, default=300.0)
    ap.add_argument('--master-time', type=float, default=120.0)
    ap.add_argument('--opp-time', type=float, default=30.0)
    ap.add_argument('--ub', type=int, default=None)
    ap.add_argument('--record', action='store_true', help='write ledger rows')
    ap.add_argument('names', nargs='+')
    A = ap.parse_args()
    for nm in A.names:
        ub = A.ub or _ub_default(nm, A.variant)
        B = Benders(nm, A.variant, ub, opp_tl=A.opp_time, log=lambda s: print(nm, s, flush=True))
        r = B.run(A.time, A.master_time)
        print(nm, A.variant, 'UB_in', ub, {k: v for k, v in r.items() if k not in ('place', 'nogoods')}, flush=True)
        os.makedirs(os.path.join(oc.ROOT, 'results', 'opp', 'benders'), exist_ok=True)
        json.dump(dict(name=nm, variant=A.variant, ub_in=ub, **{k: v for k, v in r.items() if k != 'place'}),
                  open(os.path.join(oc.ROOT, 'results', 'opp', 'benders', f'{nm}_{A.variant}.json'), 'w'))
        if A.record:
            import ledger
            sol_file = ''; checked = 0
            if r.get('place'):
                sol_file = ledger.sol_relpath(nm, A.variant, 'opp_benders', r['ub'])
                json.dump(dict(name=nm, variant=A.variant, place=r['place']), open(os.path.join(oc.ROOT, sol_file), 'w'))
                import subprocess
                out = subprocess.run([sys.executable, os.path.join(oc.ROOT, 'src', 'check_solution.py'),
                                      os.path.join(oc.ROOT, 'data', 'instances', nm + '.txt'),
                                      os.path.join(oc.ROOT, sol_file), A.variant], capture_output=True, text=True).stdout
                checked = int(out.startswith('valid') and out.strip().endswith(f'cost {r["ub"]}'))
                print(nm, 'check:', out.strip(), flush=True)
            lb = r.get('lb')
            ledger.record(nm, A.variant, 'opp_benders', r['status'], lb=lb, ub=r.get('ub'),
                          time_s=r['time'], threads=1, sol_file=sol_file, checked=checked,
                          note=f'logic Benders (CP-SAT master + DFF rows + OPP no-goods); UB_in {ub}; iters {r["iters"]}; nogoods {len(r["nogoods"])}; dff rows {r["dff_rows"]}')
