"""Upper bounds from Florian Fontan's packingsolver (pip install packingsolver; MIT licence).

Ported from the 2026-10-13 probe ('psrun.py'). In a 60 s test on MB_C3_11 (area variant) it
returned 9556, worse than the literature BKS 9404 and the CP-SAT optimum 9208, so it is a
fallback UB generator only.

usage: python packingsolver_ub.py <weighted|area> <time_s> MB_C1_1 [...]
"""
import os
import sys
import time

import packingsolver.rectangle as psr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402

mode = sys.argv[1]; tl = float(sys.argv[2])
for nm in sys.argv[3:]:
    bins, items = io.read(nm, area=(mode == 'area'))
    b = psr.InstanceBuilder()
    b.set_objective(psr.Objective.VariableSizedBinPacking)
    for W, H, c in bins:
        b.add_bin_type(W, H, cost=c, copies=len(items))
    for w, h in items:
        b.add_item_type(w, h, oriented=True, copies=1)
    inst = b.build()
    p = psr.OptimizeParameters(); p.time_limit = tl; p.verbosity_level = 0
    t0 = time.time()
    out = psr.optimize(inst, p)
    s = out.solution
    print(nm, mode, io.bk[nm][0], io.bk[nm][1], 'ps', s.cost() if s.feasible() else None,
          out.variable_sized_bin_packing_bound, round(time.time() - t0, 1), flush=True)
