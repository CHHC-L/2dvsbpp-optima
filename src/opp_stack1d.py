"""opp track: exact 1D reduction for 'stack-only' instances + independent HiGHS MIP cross-check.

If, for a bin type t, every pair of items that both fit into t satisfies w_i + w_j > W_t, then no
two items can be side by side (their x-projections must overlap), so all items of a bin of type t
are pairwise vertically separated: the bin is feasible iff sum h_i <= H_t (stack them).
Symmetrically if every pair has h_i + h_j > H_t (sum w_i <= W_t).  If EVERY bin type is of one of
these two kinds, the 2DVSBPP instance is exactly a 1D variable-sized bin packing with item
eligibility, solved here as a MIP with HiGHS (a different solver and model than the CP-SAT based
methods).  Uses highspy only (do not import ortools in this process).

usage: python src/opp_stack1d.py --variant area MB_C8_1 [...]   (prints status, obj, bound)
       python src/opp_stack1d.py --scan        (which instances are stack-only)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402


def kinds(bins, items):
    out = []
    for (W, H, c) in bins:
        fit = [(w, h) for w, h in items if w <= W and h <= H]
        vert = all(fit[a][0] + fit[b][0] > W for a in range(len(fit)) for b in range(a + 1, len(fit)))
        hori = all(fit[a][1] + fit[b][1] > H for a in range(len(fit)) for b in range(a + 1, len(fit)))
        out.append('V' if vert else ('H' if hori else None))
    return out


def solve(name, variant, tl=120.0, ub=None):
    import highspy
    import numpy as np
    bins, items = oc.load(name)
    kd = kinds(bins, items)
    if any(k is None for k in kd):
        return dict(status='NOT_STACK', kinds=kd)
    n = len(items)
    h = highspy.Highs(); h.setOptionValue('output_flag', False)
    h.setOptionValue('time_limit', float(tl)); h.setOptionValue('threads', 1)
    h.setOptionValue('mip_rel_gap', 0.0); h.setOptionValue('mip_abs_gap', 0.0)
    inf = highspy.kHighsInf
    slots = []
    for t, (W, H, c) in enumerate(bins):
        cost = oc.cost_of((W, H, c), variant)
        fit = [i for i in range(n) if items[i][0] <= W and items[i][1] <= H]
        K = len(fit) if ub is None else min(len(fit), ub // cost)
        for k in range(K):
            slots.append((t, k, fit, cost, H if kd[t] == 'V' else W))
    # variables: y_s, a_{i,s}
    var = 0; yidx = []; aidx = {}
    costs = []
    for s, (t, k, fit, cost, cap) in enumerate(slots):
        yidx.append(var); costs.append(cost); var += 1
        for i in fit:
            aidx[i, s] = var; costs.append(0.0); var += 1
    h.addVars(var, np.zeros(var), np.ones(var))
    h.changeColsCost(var, np.arange(var, dtype=np.int32), np.array(costs, dtype=float))
    h.changeColsIntegrality(var, np.arange(var, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * var))
    for i in range(n):
        idx = [aidx[i, s] for s in range(len(slots)) if (i, s) in aidx]
        h.addRow(1.0, 1.0, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
    for s, (t, k, fit, cost, cap) in enumerate(slots):
        kdim = 1 if kd[t] == 'V' else 0
        idx = [aidx[i, s] for i in fit] + [yidx[s]]
        val = [float(items[i][kdim]) for i in fit] + [-float(cap)]
        h.addRow(-inf, 0.0, len(idx), np.array(idx, dtype=np.int32), np.array(val))
    # symmetry: slots of the same type used in order
    for s in range(len(slots) - 1):
        if slots[s][0] == slots[s + 1][0]:
            h.addRow(-inf, 0.0, 2, np.array([yidx[s + 1], yidx[s]], dtype=np.int32), np.array([1.0, -1.0]))
    t0 = time.time()
    h.run()
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    return dict(status=st, obj=info.objective_function_value, bound=info.mip_dual_bound,
                gap=info.mip_gap, kinds=kd, time=time.time() - t0, slots=len(slots))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['weighted', 'area'], default='area')
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--time', type=float, default=120)
    ap.add_argument('names', nargs='*')
    A = ap.parse_args()
    if A.scan:
        import vsbpp_io as io
        hits = []
        for nm in io.all_names():
            b, it = oc.load(nm)
            if all(k is not None for k in kinds(b, it)):
                hits.append(nm)
        print(len(hits), 'stack-only instances:', ' '.join(hits))
    for nm in A.names:
        print(nm, A.variant, solve(nm, A.variant, A.time), flush=True)
