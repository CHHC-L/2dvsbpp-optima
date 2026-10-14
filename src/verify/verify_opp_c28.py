"""verify_opp: independent check of the MB_C2_8 (area) LB = 594 (no opp code; raw file read).
1. enumerate every bin multiset with area cost in [item area, 593];
2. for each, the half-height counting argument: in every vertical line the crossing items are stacked,
   sum h <= H  =>  sum floor(h/2) <= floor(H/2); integrate over x: sum w*floor(h/2) <= W*floor(H/2).
   (and the transposed version with floor(w/2) along W). A violated inequality proves infeasibility."""
import itertools, os
P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = open(f'{P}/data/instances/MB_C2_8.txt').read().split('\n'); n, m, _ = map(int, L[0].split())
B = [tuple(map(int, L[1 + t].split())) for t in range(m)]
I = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
area = sum(w * h for w, h in I); cost = [b[0] * b[1] for b in B]
vecs = []
for cnt in itertools.product(*[range(0, 594 // c + 1) for c in cost]):
    tot = sum(a * c for a, c in zip(cnt, cost))
    if area <= tot <= 593: vecs.append((tot, cnt))
print('item area', area, 'bin costs', cost, 'vectors with cost in [area, 593]:', vecs)
for tot, cnt in vecs:
    assert sum(cnt) == 1, 'multi-bin vector: needs a different argument'
    t = cnt.index(1); W, H = B[t][0], B[t][1]
    print('bin', W, 'x', H, 'all items fit individually:', all(w <= W and h <= H for w, h in I))
    lhs = sum(w * (h // 2) for w, h in I); rhs = W * (H // 2)
    lhs2 = sum(h * (w // 2) for w, h in I); rhs2 = H * (W // 2)
    print('sum w*floor(h/2) =', lhs, '> W*floor(H/2) =', rhs, ':', lhs > rhs,
          '| transposed', lhs2, rhs2, lhs2 > rhs2)
