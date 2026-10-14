"""opp track: independent HiGHS check of the opp_benders INITIAL master relaxation (no no-goods):
per-type slots, assignment, area rows, initial half-size DFF rows (exhaustively validated),
pairwise conflicts, slot order / non-empty, cost <= U-1.  If HiGHS reports Infeasible the LB U
holds by a different solver than CP-SAT.  (Do not import ortools in this process.)

usage: python src/verify/opp_master_highs.py area MB_C5_13 131765
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opp_benders as ob  # noqa: E402  (imports no ortools at module level)


def check(name, variant, U, tl=120):
    import highspy
    B = ob.Benders(name, variant, U, log=lambda s: None)  # cost <= U-1
    items = B.items; n = len(items)
    h = highspy.Highs(); h.setOptionValue('output_flag', False)
    h.setOptionValue('time_limit', float(tl)); h.setOptionValue('threads', 1)
    inf = highspy.kHighsInf
    cols = {}; cost = []

    def var(key, c=0.0):
        cols[key] = len(cost); cost.append(c); return cols[key]
    slots = []
    for q, (traw, (W, H), c) in enumerate(B.types):
        fit = [i for i in range(n) if items[i][0] <= W and items[i][1] <= H]
        K = min(len(fit), (U - 1) // c)
        for k in range(K):
            var(('u', q, k), float(c))
            for i in fit:
                if i >= k:
                    var(('a', i, q, k))
            slots.append((q, k, W, H, [i for i in fit if i >= k]))
    nv = len(cost)
    h.addVars(nv, np.zeros(nv), np.ones(nv))
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), np.array(cost))
    h.changeColsIntegrality(nv, np.arange(nv, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * nv))

    def row(lo, hi, idx, val):
        h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=float))
    for i in range(n):
        idx = [v for key, v in cols.items() if key[0] == 'a' and key[1] == i]
        row(1, 1, idx, [1] * len(idx))
    for q, k, W, H, its in slots:
        u = cols['u', q, k]; A = [cols['a', i, q, k] for i in its]
        row(-inf, 0, A + [u], [items[i][0] * items[i][1] for i in its] + [-W * H])
        row(0, inf, A + [u], [1] * len(A) + [-1])          # used slot non-empty
        for ft, gt, cap, lab in B.dff_rows.get(q, []):
            row(-inf, 0, A + [u], [ft[items[i][0]] * gt[items[i][1]] for i in its] + [-cap])
        for x1 in range(len(its)):
            for x2 in range(x1 + 1, len(its)):
                i, j = its[x1], its[x2]
                if items[i][0] + items[j][0] > W and items[i][1] + items[j][1] > H:
                    row(-inf, 0, [cols['a', i, q, k], cols['a', j, q, k], u], [1, 1, -1])
        if k > 0:
            row(-inf, 0, [u, cols['u', q, k - 1]], [1, -1])
    uk = [v for key, v in cols.items() if key[0] == 'u']
    row(-inf, U - 1, uk, [cost[v] for v in uk])
    h.run()
    return h.modelStatusToString(h.getModelStatus()), nv


if __name__ == '__main__':
    var_, nm, U = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(nm, var_, 'cost <=', U - 1, check(nm, var_, U), flush=True)
