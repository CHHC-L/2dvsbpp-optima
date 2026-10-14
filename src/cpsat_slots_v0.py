"""Earlier, independent CP-SAT formulation (first scouting probe, 2026-10-13, 'probe/vsbpp.py').

Weighted variant only. Generic slots k = 0..n-1 with a bin-type choice u[k][t] per slot (instead of
per-type slots as in cpsat_direct.py); item i may only use slots k <= i. It is WEAKER than
cpsat_direct.py on 40-item instances (e.g. MB_C4_11: 9783 vs proven 6934 after 60 s) but it is a
genuinely different model, so it serves as a cross-check: on all 100 20-item instances both models
prove the same weighted optima.

Originally read the OR-Datasets JSON copy; now reads data/instances via vsbpp_io (the JSON copy
merges duplicate bin-type lines, which is harmless because bin supply is unlimited).

usage: python cpsat_slots_v0.py <time_s> MB_C1_1 [...]
"""
import os
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402


def solve(nm, tl, workers=4):
    bins, items = io.read(nm, area=False)
    bins = sorted(set(bins))  # merge duplicate bin-type lines (unlimited supply)
    n = len(items); T = len(bins)
    items = sorted(items, key=lambda r: -r[0] * r[1])
    K = n
    Wmax = max(b[0] for b in bins); Hmax = max(b[1] for b in bins)
    m = cp_model.CpModel()
    u = [[m.NewBoolVar(f"u{k}_{t}") for t in range(T)] for k in range(K)]
    used = [m.NewBoolVar(f"used{k}") for k in range(K)]
    for k in range(K):
        m.Add(sum(u[k]) == used[k])
    Wk = [m.NewIntVar(0, Wmax, f"W{k}") for k in range(K)]
    Hk = [m.NewIntVar(0, Hmax, f"H{k}") for k in range(K)]
    for k in range(K):
        m.Add(Wk[k] == sum(bins[t][0] * u[k][t] for t in range(T)))
        m.Add(Hk[k] == sum(bins[t][1] * u[k][t] for t in range(T)))
    a = {}
    for i, (w, h) in enumerate(items):
        lits = []
        for k in range(min(i + 1, K)):
            a[i, k] = m.NewBoolVar(f"a{i}_{k}")
            lits.append(a[i, k])
            m.AddImplication(a[i, k], used[k])
            for t in range(T):
                if w > bins[t][0] or h > bins[t][1]:
                    m.AddImplication(a[i, k], u[k][t].Not())
        m.AddExactlyOne(lits)
    for k in range(1, K):
        m.AddImplication(used[k], used[k - 1])
    for k in range(K):
        xi = []; yi = []; area = []
        for i, (w, h) in enumerate(items):
            if (i, k) not in a:
                continue
            x = m.NewIntVar(0, Wmax - w, f"x{i}_{k}"); y = m.NewIntVar(0, Hmax - h, f"y{i}_{k}")
            m.Add(x + w <= Wk[k]).OnlyEnforceIf(a[i, k]); m.Add(y + h <= Hk[k]).OnlyEnforceIf(a[i, k])
            xi.append(m.NewOptionalFixedSizeIntervalVar(x, w, a[i, k], f"xi{i}_{k}"))
            yi.append(m.NewOptionalFixedSizeIntervalVar(y, h, a[i, k], f"yi{i}_{k}"))
            area.append(w * h * a[i, k])
        if xi:
            m.AddNoOverlap2D(xi, yi)
            m.Add(sum(area) <= sum(bins[t][0] * bins[t][1] * u[k][t] for t in range(T)))
    m.Minimize(sum(bins[t][2] * u[k][t] for k in range(K) for t in range(T)))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
    t0 = time.time(); st = s.Solve(m)
    obj = s.ObjectiveValue() if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    return nm, n, T, s.StatusName(st), obj, s.BestObjectiveBound(), round(time.time() - t0, 1)


if __name__ == "__main__":
    tl = float(sys.argv[1])
    for nm in sys.argv[2:]:
        print(*solve(nm, tl), flush=True)
