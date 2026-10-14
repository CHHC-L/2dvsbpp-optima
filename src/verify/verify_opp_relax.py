"""verify_opp: INDEPENDENT re-implementation (no opp_* code imported) of two LB arguments.

mode 'relax'  : bin-slot assignment relaxation with only textbook-valid rows:
                per-slot area; 'tall' items (2h > H) pairwise cannot stack -> sum w <= W;
                'wide' items (2w > W) -> sum h <= H; pairwise conflicts (w_i+w_j>W and h_i+h_j>H).
                Symmetry: used slots of a type are 0..r-1 (u_{k+1} <= u_k) and slot k only holds
                items with index >= k (relabel bins by smallest item index; valid for any fixed order).
                Solved for MIN cost (solver: highs or cpsat). Optimum = valid LB for the 2DVSBPP.
mode 'stack1d': if every pair of items is too wide to sit side by side in every bin type that holds
                both, the problem is exactly 1D (sum h <= H per bin); solve exactly with CP-SAT/HiGHS.
Raw file reading as in check_solution.py: line 1..m: W H 1 cost; items: idx w h 1; w along W.
usage: python src/verify/verify_opp_relax.py <relax|stack1d> <highs|cpsat> <name> <area|weighted> [tl]
"""
import sys, time, os
mode, solver, name, var = sys.argv[1:5]
tl = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = open(f'{P}/data/instances/{name}.txt').read().split('\n')
n, m, _ = map(int, L[0].split())
B = [tuple(map(int, L[1 + t].split())) for t in range(m)]
I = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
cost = [(b[0] * b[1] if var == 'area' else b[3]) for b in B]
UBcap = int(sys.argv[6]) if len(sys.argv) > 6 else None   # optional: only cost <= UBcap
assert all(c > 0 for c in cost)
if mode == 'stack1d':
    bad = [(i, j, t) for t in range(m) for i in range(n) for j in range(i + 1, n)
           if I[i][0] <= B[t][0] and I[j][0] <= B[t][0] and I[i][1] <= B[t][1] and I[j][1] <= B[t][1]
           and I[i][0] + I[j][0] <= B[t][0]]
    print('pairs that could sit side by side:', len(bad), bad[:5])
    if bad:
        sys.exit(0)
# slots
cap_total = UBcap if UBcap else sum(cost[t] for t in range(m)) * n
slots = []
for t in range(m):
    W, H = B[t][0], B[t][1]
    fit = [i for i in range(n) if I[i][0] <= W and I[i][1] <= H]
    K = min(len(fit), cap_total // cost[t])
    for k in range(K):
        slots.append((t, k, [i for i in fit if i >= k]))
def dff_ok(t, C):   # exhaustive: every multiset of sizes with sum <= C has sum f <= f(C)
    best = [0] * (C + 1)
    for cap in range(1, C + 1):
        best[cap] = max([best[cap - 1]] + [t[a] + best[cap - a] for a in range(1, cap + 1)])
    return t[0] == 0 and min(t) >= 0 and best[C] <= t[C]
_DFF = {}
def dffs(C):
    if C not in _DFF:
        out = []
        for k in range(1, 9):   # Fekete-Schepers u^(k), scaled by k*C
            out.append([k * a if ((k + 1) * a) % C == 0 else C * (((k + 1) * a) // C) for a in range(C + 1)])
        for q in range(2, 7):   # CCM f0 with lambda = C // q
            lam = max(1, C // q)
            out.append([0 if a < lam else (C if a > C - lam else a) for a in range(C + 1)])
        _DFF[C] = [t for t in out if dff_ok(t, C)]
    return _DFF[C]
def dff_pairs(W, H):
    idW = list(range(W + 1)); idH = list(range(H + 1))
    return [(idW, g) for g in dffs(H)] + [(f, idH) for f in dffs(W)]
t0 = time.time()
if solver == 'cpsat':
    from ortools.sat.python import cp_model
    md = cp_model.CpModel(); u = {}; a = {}
    for (t, k, its) in slots:
        W, H = B[t][0], B[t][1]
        u[t, k] = md.NewBoolVar('')
        for i in its:
            a[i, t, k] = md.NewBoolVar(''); md.AddImplication(a[i, t, k], u[t, k])
        A = lambda f: sum(f(i) * a[i, t, k] for i in its)
        if mode == 'stack1d':
            md.Add(sum(I[i][1] * a[i, t, k] for i in its) <= H)
        else:
            md.Add(sum(I[i][0] * I[i][1] * a[i, t, k] for i in its) <= W * H)
            tall = [i for i in its if 2 * I[i][1] > H]; wide = [i for i in its if 2 * I[i][0] > W]
            if tall: md.Add(sum(I[i][0] * a[i, t, k] for i in tall) <= W)
            if wide: md.Add(sum(I[i][1] * a[i, t, k] for i in wide) <= H)
            for x in range(len(its)):
                for y in range(x + 1, len(its)):
                    i, j = its[x], its[y]
                    if I[i][0] + I[j][0] > W and I[i][1] + I[j][1] > H:
                        md.AddBoolOr([a[i, t, k].Not(), a[j, t, k].Not()])
        md.AddBoolOr([a[i, t, k] for i in its] + [u[t, k].Not()])
        if k > 0: md.AddImplication(u[t, k], u[t, k - 1])
    for i in range(n):
        md.AddExactlyOne([v for (j, t, k), v in a.items() if j == i])
    obj = sum(cost[t] * v for (t, k), v in u.items())
    if UBcap: md.Add(obj <= UBcap)
    md.Minimize(obj)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 1
    st = s.Solve(md)
    res = dict(status=s.StatusName(st), bound=s.BestObjectiveBound(),
               obj=(s.ObjectiveValue() if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None))
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res['bins'] = sorted(t for (t, k), v in u.items() if s.Value(v))
else:
    import highspy, numpy as np
    h = highspy.Highs(); h.setOptionValue('output_flag', False)
    h.setOptionValue('time_limit', tl); h.setOptionValue('threads', 1)
    h.setOptionValue('mip_rel_gap', 0.0); h.setOptionValue('mip_abs_gap', 0.0)
    col = {}; c = []
    for (t, k, its) in slots:
        col['u', t, k] = len(c); c.append(cost[t])
        for i in its:
            col['a', i, t, k] = len(c); c.append(0)
    nv = len(c); inf = highspy.kHighsInf
    h.addVars(nv, np.zeros(nv), np.ones(nv))
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), np.array(c, dtype=float))
    h.changeColsIntegrality(nv, np.arange(nv, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * nv))
    def row(lo, hi, idx, val):
        h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=float))
    for (t, k, its) in slots:
        W, H = B[t][0], B[t][1]; uu = col['u', t, k]
        for i in its: row(-inf, 0, [col['a', i, t, k], uu], [1, -1])
        if mode == 'stack1d':
            row(-inf, 0, [col['a', i, t, k] for i in its] + [uu], [I[i][1] for i in its] + [-H])
        else:
            row(-inf, 0, [col['a', i, t, k] for i in its] + [uu], [I[i][0] * I[i][1] for i in its] + [-W * H])
            tall = [i for i in its if 2 * I[i][1] > H]; wide = [i for i in its if 2 * I[i][0] > W]
            if tall: row(-inf, W, [col['a', i, t, k] for i in tall], [I[i][0] for i in tall])
            if wide: row(-inf, H, [col['a', i, t, k] for i in wide], [I[i][1] for i in wide])
            for x in range(len(its)):
                for y in range(x + 1, len(its)):
                    i, j = its[x], its[y]
                    if I[i][0] + I[j][0] > W and I[i][1] + I[j][1] > H:
                        row(-inf, 1, [col['a', i, t, k], col['a', j, t, k]], [1, 1])
        row(-inf, 0, [uu] + [col['a', i, t, k] for i in its], [1] + [-1] * len(its))
        if k > 0: row(-inf, 0, [uu, col['u', t, k - 1]], [1, -1])
        if os.environ.get('VO_DFF') and mode == 'relax':
            for ft, gt in dff_pairs(W, H):
                row(-inf, 0, [col['a', i, t, k] for i in its] + [uu], [ft[I[i][0]] * gt[I[i][1]] for i in its] + [-ft[W] * gt[H]])
    for i in range(n):
        idx = [v for key, v in col.items() if key[0] == 'a' and key[1] == i]
        row(1, 1, idx, [1] * len(idx))
    if UBcap:
        uk = [v for key, v in col.items() if key[0] == 'u']
        row(-inf, UBcap, uk, [c[v] for v in uk])
    if len(sys.argv) > 7:   # optional no-goods json: [[Wn, Hn, [[w,h],...]], ...] (multisets that do not fit Wn x Hn)
        import json, itertools
        from collections import Counter
        NG = json.load(open(sys.argv[7])); nng = 0
        for Wn, Hn, S in NG:
            need = Counter(tuple(d) for d in S)
            for (t, k, its) in slots:
                if B[t][0] > Wn or B[t][1] > Hn: continue
                choices = [list(itertools.combinations([i for i in its if I[i] == d], cnt)) for d, cnt in need.items()]
                for combo in itertools.product(*choices):
                    idx = [i for part in combo for i in part]
                    row(-inf, len(idx) - 1, [col['a', i, t, k] for i in idx], [1] * len(idx)); nng += 1
        print('no-good rows', nng, flush=True)
    h.run()
    info = h.getInfo()
    res = dict(status=h.modelStatusToString(h.getModelStatus()), bound=info.mip_dual_bound,
               obj=info.objective_function_value)
    if os.environ.get('VO_DUMP') and res['status'] == 'Optimal':
        import json
        xv = h.getSolution().col_value; binsol = {}
        for key, v in col.items():
            if key[0] == 'a' and xv[v] > 0.5:
                binsol.setdefault(f'{key[2]}_{key[3]}', []).append(list(I[key[1]]))
        json.dump(dict(name=name, bins={k: dict(W=B[int(k.split('_')[0])][0], H=B[int(k.split('_')[0])][1], items=v)
                                        for k, v in binsol.items()}), open(os.environ['VO_DUMP'], 'w'))
print(name, var, mode, solver, 'slots', len(slots), res, 'time %.1f' % (time.time() - t0), flush=True)
