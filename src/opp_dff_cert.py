"""opp track: STANDALONE checker for a dual-feasible-function (DFF) infeasibility certificate of a
single-bin 2D orthogonal packing (shares no code with the solvers; re-reads the RAW instance file).

usage: python opp_dff_cert.py <instance.txt> <cert.json>

cert.json: {"bin_line": t (0-based raw bin line), "items": "all" | [0-based raw item lines],
            "f": [f(0), ..., f(W)],  "g": [g(0), ..., g(H)]}
Checks
  1. f is a DFF on [0, W]: max over multisets of sizes in 1..W with sum <= W of sum f  <=  f(W)
     (exhaustive unbounded-knapsack DP), f(0) = 0; same for g on [0, H];
  2. sum_i f(w_i) g(h_i) > f(W) g(H)  (exact integers).
If f is the identity, validity is elementary: for every unit column [x, x+1) the items crossing it
are stacked disjointly, so the sum of their heights is <= H, hence sum of g(h) <= g(H) (DFF);
integrating over the W columns gives sum_i w_i g(h_i) <= W g(H).  For general f, g this is the
Fekete-Schepers (2004) conservative-scale bound.
Prints 'CERT OK: items do not fit' or raises.
"""
import json
import sys


def dff_ok(t, C):
    assert len(t) == C + 1 and t[0] == 0
    best = [0] * (C + 1)
    for cap in range(1, C + 1):
        b = best[cap - 1]
        for a in range(1, cap + 1):
            b = max(b, t[a] + best[cap - a])
        best[cap] = b
    return best[C] <= t[C]


inst, certf = sys.argv[1], sys.argv[2]
L = open(inst).read().split('\n'); n, m, _ = map(int, L[0].split())
bins = [tuple(map(int, L[1 + t].split())) for t in range(m)]
items = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
C = json.load(open(certf))
W, H = bins[C['bin_line']][0], bins[C['bin_line']][1]
sel = items if C['items'] == 'all' else [items[i] for i in C['items']]
f, g = C['f'], C['g']
assert all(isinstance(v, int) for v in f + g)
assert dff_ok(f, W), 'f is not a DFF'
assert dff_ok(g, H), 'g is not a DFF'
assert all(w <= W and h <= H for w, h in sel)
lhs = sum(f[w] * g[h] for w, h in sel)
rhs = f[W] * g[H]
ident = f == list(range(W + 1))
print(f'bin {W}x{H}  items {len(sel)}  area {sum(w * h for w, h in sel)} / {W * H}  '
      f'f identity: {ident}  lhs {lhs}  rhs {rhs}')
assert lhs > rhs, 'certificate does not prove infeasibility'
print('CERT OK: items do not fit')
