"""opp track: SAT order encoding of single-bin 2D orthogonal packing (Soh, Inoue, Tamura,
Banbara & Nabeshima 2010) with DRAT certificates (CaDiCaL binary -> drat-trim).

Variables (1-based DIMACS):
  px[i][e]  <->  x_i <= e   (e = 0 .. W-w_i-1;  x_i <= W-w_i is the constant TRUE)
  py[i][f]  <->  y_i <= f
  lr[i][j]  ->   x_i + w_i <= x_j          ud[i][j] -> y_i + h_i <= y_j
Clauses: order axioms; for every pair i<j  lr_ij v lr_ji v ud_ij v ud_ji; the implications
  lr_ij -> (x_j <= v -> x_i <= v - w_i) for every v; same for ud.
Optional validity-safe reductions (same justification as opp_cpsat.py):
  normal  : x_i not in its normal-pattern set is forbidden (clause  [x<=v] -> [x<=v-1])
  reflect : one unique item k: x_k <= (W-w_k)//2, y_k <= (H-h_k)//2
  ident   : identical items i<j: x_i <= x_j and NOT lr_ji
  (the pairwise rule 'NOT lr_ji and ud_ji -> lr_ij' of the literature is NOT used: it can cut
   all packings of 3+ identical items: 2x2 squares at (0,4), (2,0), (1,2) admit no valid labelling)
The certificate is for the CNF as written: the CNF file + DRAT proof + drat-trim 'VERIFIED' show
the CNF unsatisfiable; correctness of the encoding is validated by positive/negative controls
and a randomised CP-SAT cross-check (src/verify/opp_selftest.py).

usage: python src/opp_sat.py MB_C2_8 27 19 [--plain] [--out results/opp/certs]
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402

TOOLS = os.path.join(oc.ROOT, 'third_party')  # built by scripts/get_sat_tools.sh; else taken from PATH
CADICAL = os.path.join(TOOLS, 'cadical', 'build', 'cadical')
DRATTRIM = os.path.join(TOOLS, 'drat-trim', 'drat-trim')
if not os.path.exists(CADICAL):
    CADICAL = shutil.which('cadical') or CADICAL
if not os.path.exists(DRATTRIM):
    DRATTRIM = shutil.which('drat-trim') or DRATTRIM


def encode(items, W, H, normal=True, reflect=True, ident=True):
    n = len(items)
    nv = [0]

    def new():
        nv[0] += 1
        return nv[0]
    px = [[new() for e in range(W - w)] for (w, h) in items]
    py = [[new() for f in range(H - h)] for (w, h) in items]
    lr = {}; ud = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                lr[i, j] = new(); ud[i, j] = new()
    cl = []
    for i, (w, h) in enumerate(items):
        if w > W or h > H:
            cl.append([])
            return nv[0], cl, px, py
        for e in range(W - w - 1):
            cl.append([-px[i][e], px[i][e + 1]])
        for f in range(H - h - 1):
            cl.append([-py[i][f], py[i][f + 1]])

    def le(P, i, v, dom_max):
        """literal for [z_i <= v]: returns True / False / literal."""
        if v < 0:
            return False
        if v >= dom_max:
            return True
        return P[i][v]

    def sep(P, rel, sz, C):
        # rel[i,j] -> z_i + sz_i <= z_j
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = rel[i, j]
                dj = C - sz[j]; di = C - sz[i]
                for v in range(0, dj + 1):
                    c = [-r]
                    a = le(P, j, v, dj)  # [z_j <= v]
                    if a is True:
                        pass
                    else:
                        c.append(-a)
                    b = le(P, i, v - sz[i], di)
                    if b is True:
                        continue
                    if b is not False:
                        c.append(b)
                    cl.append(c)
    ws = [w for w, h in items]; hs = [h for w, h in items]
    sep(px, lr, ws, W)
    sep(py, ud, hs, H)
    for i in range(n):
        for j in range(i + 1, n):
            cl.append([lr[i, j], lr[j, i], ud[i, j], ud[j, i]])
    if normal:
        for i in range(n):
            for P, sz, C in ((px, ws, W), (py, hs, H)):
                d = C - sz[i]
                N = set(oc.normal_patterns(sz, i, d))
                for v in range(d + 1):
                    if v in N:
                        continue
                    # forbid z_i = v :  [z<=v] -> [z<=v-1]
                    c = []
                    a = le(P, i, v, d)
                    if a is not True:
                        c.append(-a)
                    b = le(P, i, v - 1, d)
                    if b is True:
                        continue
                    if b is not False:
                        c.append(b)
                    cl.append(c)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it, []).append(i)
    if reflect:
        uniq = [g[0] for g in groups.values() if len(g) == 1]
        if uniq:
            k = max(uniq, key=lambda i: (items[i][0] * items[i][1], i))
            for P, sz, C in ((px, ws, W), (py, hs, H)):
                v = (C - sz[k]) // 2
                if v < C - sz[k]:
                    cl.append([P[k][v]])
    if ident:
        for g in groups.values():
            for a_, b_ in zip(g, g[1:]):
                d = W - ws[a_]
                for v in range(d):
                    cl.append([-px[b_][v], px[a_][v]])
                cl.append([-lr[b_, a_]])
    return nv[0], cl, px, py


def write_cnf(path, nv, cl, comment=''):
    with open(path, 'w') as f:
        if comment:
            for line in comment.split('\n'):
                f.write('c ' + line + '\n')
        f.write(f'p cnf {nv} {len(cl)}\n')
        for c in cl:
            f.write(' '.join(map(str, c)) + ' 0\n')


def decode(model, items, W, H, px, py):
    val = set(l for l in model if l > 0)
    pl = []
    for i, (w, h) in enumerate(items):
        x = next((e for e in range(W - w) if px[i][e] in val), W - w)
        y = next((f for f in range(H - h) if py[i][f] in val), H - h)
        pl.append((w, h, x, y))
    return pl


def solve(items, W, H, workdir, tag, tl=600, normal=True, reflect=True, ident=True, check=True):
    """returns dict(status=SAT/UNSAT/UNKNOWN, placement, cnf, proof, drat='VERIFIED'/...)"""
    os.makedirs(workdir, exist_ok=True)
    nv, cl, px, py = encode(items, W, H, normal, reflect, ident)
    cnf = os.path.join(workdir, f'{tag}.cnf')
    proof = os.path.join(workdir, f'{tag}.drat')
    write_cnf(cnf, nv, cl, f'opp order encoding {tag} W={W} H={H} normal={normal} reflect={reflect} ident={ident}\n'
                           f'items {items}')
    t0 = time.time()
    p = subprocess.run([CADICAL, '-q', '-t', str(int(tl)), '--binary=false', cnf, proof],
                       capture_output=True, text=True)
    dt = time.time() - t0
    out = p.stdout
    res = dict(vars=nv, clauses=len(cl), cnf=cnf, proof=proof, sat_time=dt)
    if p.returncode == 10:
        model = [int(t) for line in out.split('\n') if line.startswith('v') for t in line.split()[1:]]
        pl = decode(model, items, W, H, px, py)
        ok = oc.verify_place(items, {0: (W, H)}, [(0, w, h, x, y) for w, h, x, y in pl])
        res.update(status='SAT' if ok else 'SAT_DECODE_FAIL', placement=pl)
        os.remove(proof)
    elif p.returncode == 20:
        res.update(status='UNSAT')
        if check:
            t1 = time.time()
            q = subprocess.run([DRATTRIM, cnf, proof, '-t', str(int(max(tl, 600)))], capture_output=True, text=True)
            res['drat'] = 'VERIFIED' if 's VERIFIED' in q.stdout else ('NOT VERIFIED: ' + q.stdout[-300:])
            res['drat_time'] = time.time() - t1
    else:
        res.update(status='UNKNOWN')
    return res


def gzip_file(path):
    with open(path, 'rb') as f, gzip.open(path + '.gz', 'wb') as g:
        shutil.copyfileobj(f, g)
    os.remove(path)
    return path + '.gz'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('W', type=int); ap.add_argument('H', type=int)
    ap.add_argument('--plain', action='store_true', help='no normal patterns / symmetry breaking')
    ap.add_argument('--time', type=float, default=600)
    ap.add_argument('--out', default=os.path.join(oc.ROOT, 'results', 'opp', 'certs'))
    A = ap.parse_args()
    bins, items = oc.load(A.name)
    tag = f'{A.name}_{A.W}x{A.H}' + ('_plain' if A.plain else '_sym')
    r = solve(items, A.W, A.H, A.out, tag, tl=A.time, normal=not A.plain, reflect=not A.plain, ident=not A.plain)
    sz = os.path.getsize(r['proof']) if os.path.exists(r['proof']) else 0
    print(tag, {k: v for k, v in r.items() if k != 'placement'}, 'proof bytes', sz, flush=True)
    if r['status'] == 'UNSAT' and sz < 50e6:
        gzip_file(r['proof']); gzip_file(r['cnf'])
