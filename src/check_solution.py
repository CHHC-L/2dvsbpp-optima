"""Standalone solution checker. Re-reads the RAW instance file; shares no code with the solvers.

usage: python check_solution.py <instance.txt> <solution.json> <weighted|area>

solution.json: {"place": [[t, k, w, h, x, y], ...], ...}
  t = bin-type index (0-based line order in the instance file), k = copy index of that type,
  (w, h) = item dimensions (no rotation), (x, y) = lower-left corner.
All six entries must be non-negative integers.
The reported cost uses column 4 of the bin line ('weighted') or W*H ('area').
Exit status 0 and a line 'valid; bins used [...] cost C' if the packing is valid; otherwise an
AssertionError (non-zero exit status).
"""
import json
import sys

inst, solf, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ('weighted', 'area')
L = open(inst).read().split('\n'); n, m, _ = map(int, L[0].split())
bins = [tuple(map(int, L[1 + t].split())) for t in range(m)]
items = sorted(tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n))
S = json.load(open(solf)); place = S['place']
for p in place:
    assert len(p) == 6 and all(type(v) is int and v >= 0 for v in p), f'bad entry {p}'
    assert p[0] < m, f'bin type {p[0]} out of range (m = {m})'
assert sorted((p[2], p[3]) for p in place) == items, 'item multiset mismatch'
used = {}
for t, k, w, h, x, y in place:
    W, H = bins[t][0], bins[t][1]
    assert x + w <= W and y + h <= H, 'outside'
    used.setdefault((t, k), []).append((x, y, w, h))
for key, rs in used.items():
    for a in range(len(rs)):
        for b in range(a + 1, len(rs)):
            xa, ya, wa, ha = rs[a]; xb, yb, wb, hb = rs[b]
            assert not (xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha), 'overlap'
cost = sum((bins[t][0] * bins[t][1] if mode == 'area' else bins[t][3]) for (t, k) in used)
print('valid; bins used', sorted(t for (t, k) in used), 'cost', cost)
