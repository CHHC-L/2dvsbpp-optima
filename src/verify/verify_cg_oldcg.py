"""verify_cg: recompute LBs with the OLD probe CG code (src/cg_bound.py: CP-SAT pricing, Farley bound,
lattice rounding; no DP, no dominance, no IP rounding), forced to 1 CP-SAT worker, no RMP.
usage: python src/verify/verify_cg_oldcg.py HH:MM tl_total name:variant:claimedLB ..."""
import json, os, sys, time
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SRC); sys.path.insert(0, SRC); os.chdir(ROOT)
import numpy as np
import cg_bound as B
import vsbpp_io as io
import ledger
_orig = B.cp_model.CpSolver
class _One(_orig):
    def Solve(self, m, *a, **k):
        self.parameters.num_workers = 1
        return super().Solve(m, *a, **k)
B.cp_model.CpSolver = _One
hh, mm = map(int, sys.argv[1].split(':')); tlt = float(sys.argv[2])
lt = time.localtime(); stop_ts = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
for arg in sys.argv[3:]:
    nm, var, L = arg.split(':'); L = int(L)
    left = stop_ts - time.time()
    if left < 10: print('stop reached', flush=True); break
    tl = min(tlt, left - 5)
    bins, items = io.read(nm, var == 'area'); n = len(items); t0 = time.time()
    cols = []; costs = []; seen = set()
    for i, (w, h) in enumerate(items):
        c, t = min((c, t) for t, (W, H, c) in enumerate(bins) if w <= W and h <= H)
        cols.append(frozenset([i])); costs.append(c); seen.add((t, cols[-1]))
    best = 0.0; conv = False
    while time.time() - t0 < tl:
        z, duals, _ = B.solve_lp(cols, costs, n); theta = 1.0; added = 0
        for t, (W, H, c) in enumerate(bins):
            out, pub = B.price(items, W, H, duals, min(5.0, max(0.5, tl - (time.time() - t0))))
            if pub > c + 1e-9: theta = min(theta, c / pub)
            for val, S in out:
                if val > c + 1e-7 and (t, S) not in seen:
                    seen.add((t, S)); cols.append(S); costs.append(c); added += 1
        best = max(best, theta * float(np.sum(duals)))
        if added == 0: conv = True; break
    cap = L + max(c for _, _, c in bins) + 1
    lr = B.lattice_roundup(best, [c for _, _, c in bins], cap)
    v = 'CONFIRMED' if lr >= L else 'NOT_REACHED'
    print(json.dumps(dict(name=nm, variant=var, claimed=L, old_lb=round(best, 3), old_round=lr, conv=conv, verdict=v, t=round(time.time() - t0, 1))), flush=True)
    ledger.record(nm, var, 'verify_cg_oldcg', 'LB_ONLY', lb=lr, time_s=time.time() - t0, threads=1,
                  note=f'old cg_bound.py CG (1 worker) Farley+lattice; conv={conv}; claimed cg LB {L}: {v}')
