"""confirm_closures: second-method confirmation of closed pairs (follow-up of the 2026-10-14 sprint).

A pair (name, variant) is *closed* in results/final/results.csv when the best LB equals the cost UB
of a re-checked packing, and *confirmed* when two LB sources from disjoint code-base families reach
UB. For every closed-but-unconfirmed pair this driver tries to prove OPT >= UB, i.e. that NO packing
of cost <= UB-1 exists, with a method whose family is NOT among the row's lb_families.
No method is seeded with a ledger LB (no use_ledger_lb, no lb_ext, no 'lo' argument), so each proof
stands on its own code; the only ledger value any method reads is the checked UB (as a cap).

method         call                                                 proof                    ledger id (family)
  rep            sweep_rep.solve(ub=UB-1)                             INFEASIBLE               confirm_rep (rep)
  direct         cpsat_direct.solve(ub=UB-1)                          INFEASIBLE               confirm_direct (direct)
  lattice        lattice_exact.run(ub=UB-1, record=False)             every bin-count vector   confirm_lattice (lattice)
                 (ub given explicitly => vectors of cost <= UB-1)     of cost <= UB-1 refuted
  lattice_indep  verify_lattice_indep.py <nm> <v> UB tot vec workers  RESULT ... CONFIRMED     confirm_lattice_indep
                 (NONEMPTY=1 SORTITEMS=1, no 'lo'; subprocess)                                 (lattice_indep)
  cg             cg_bendex.py (subprocess, --out results/confirm/)    its LB >= UB             cg_v2_bendex (cg),
                                                                                               row written by cg_bendex
A successful proof is recorded as status LB_ONLY with lb = UB; a run that proves nothing as UNKNOWN
without lb (cg_bendex writes its own row with whatever LB it reached). A packing of cost <= UB-1
found by any method would REFUTE the closure: it is stored under results/solutions/sprint/, checked
with src/check_solution.py, recorded with its ub, results/confirm/REFUTATION_FOUND is created and
every shard stops.

Methods are tried per pair cheapest first, skipping the pair's own families:
  [rep, cg, lattice, direct] minus lb_families; lattice_indep takes lattice's place when lattice is
  excluded. A (pair, method) already run with a time limit >= --time (results/confirm/confirm.jsonl)
  is skipped; a pair is skipped once any run confirmed it.

usage:
  .venv/bin/python src/verify/confirm_closures.py --plan
  .venv/bin/python src/verify/confirm_closures.py --time 60 --workers 2 [--vec-time 20] [--shard 0/3]
        [--methods rep,cg] [--stop-at HH:MM] [name:variant ...]
  default pairs: every closed=1, confirmed=0 row of results/final/results.csv; a pair closed after that
  table was built can be given explicitly as name:variant:UB:fam1+fam2 (its current lb families).
  One JSON line per run is appended to results/confirm/confirm.jsonl.
"""
import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import ledger  # noqa: E402
import vsbpp_io as io  # noqa: E402

ROOT = io.ROOT
OUT = os.path.join(ROOT, 'results', 'confirm')
JSONL = os.path.join(OUT, 'confirm.jsonl')
FLAG = os.path.join(OUT, 'REFUTATION_FOUND')
RESULTS = os.path.join(ROOT, 'results', 'final', 'results.csv')
MID = {'rep': 'confirm_rep', 'direct': 'confirm_direct', 'lattice': 'confirm_lattice',
       'lattice_indep': 'confirm_lattice_indep', 'cg': 'cg_v2_bendex'}
BASE_ORDER = ['rep', 'cg', 'lattice', 'direct']
MAX_VECTORS = 1_500_000   # lattice enumerations larger than this are skipped (memory / time)


def order_for(fams):
    out = []
    for m in BASE_ORDER:
        if m == 'lattice' and 'lattice' in fams:
            if 'lattice_indep' not in fams:
                out.append('lattice_indep')
            continue
        if m not in fams:
            out.append(m)
    return out


def open_pairs():
    """closed but unconfirmed rows of results/final/results.csv."""
    out = []
    for r in csv.DictReader(open(RESULTS)):
        if r['closed'] == '1' and r['confirmed'] == '0':
            out.append(dict(name=r['name'], variant=r['variant'], n=int(r['n']), ub=int(r['ub']),
                            fams=set(r['lb_families'].split()), lb_methods=r['lb_methods']))
    return out


def history():
    return [json.loads(l) for l in open(JSONL) if l.strip()] if os.path.exists(JSONL) else []


def count_vectors(costs, caps, vmax, nmax, limit=MAX_VECTORS):
    """number of count vectors (0 <= k_t <= caps[t], sum k_t <= nmax, cost <= vmax, not all 0),
    stopping early once it exceeds `limit`."""
    T = len(costs)
    tot = [0]

    def dfs(t, cost, cnt):
        if tot[0] > limit:
            return
        c = costs[t]
        kmax = min(caps[t], (vmax - cost) // c, nmax - cnt)
        if t == T - 1:
            tot[0] += kmax + 1
            return
        for k in range(kmax + 1):
            dfs(t + 1, cost + k * c, cnt + k)
    dfs(0, 0, 0)
    return tot[0] - 1


def lattice_size(nm, variant, ub, indep=False):
    bins, items = io.read(nm, variant == 'area')
    n = len(items)
    if indep:   # verify_lattice_indep: all raw lines, only cost and count limits
        return count_vectors([c for _, _, c in bins], [n] * len(bins), ub - 1, n)
    import lattice_exact as LX
    types = LX.reduce_types(bins)
    caps = [min(sum(1 for it in items if LX.fits(it, ty)), (ub - 1) // ty[2], n) for ty in types]
    return count_vectors([ty[2] for ty in types], caps, ub - 1, n)


# ----------------------------------------------------------------------------- refutation path
def store_and_check(nm, variant, mid, place, cost):
    rel = ledger.sol_relpath(nm, variant, mid, cost)
    os.makedirs(os.path.dirname(os.path.join(ROOT, rel)), exist_ok=True)
    json.dump(dict(name=nm, variant=variant, method=mid, cost=int(cost),
                   place=[[int(v) for v in p] for p in place]), open(os.path.join(ROOT, rel), 'w'))
    return rel, recheck(nm, variant, rel, cost)


def recheck(nm, variant, rel, cost):
    p = subprocess.run([sys.executable, os.path.join(SRC, 'check_solution.py'), io.instance_path(nm),
                        os.path.join(ROOT, rel), variant], capture_output=True, text=True)
    out = p.stdout.strip()
    return p.returncode == 0 and out.startswith('valid') and out.endswith(f'cost {int(cost)}')


def refuted(nm, variant, mid, ub, cost, rel, ok, dt, workers, how):
    msg = (f'!!! REFUTATION: {nm} {variant}: {how} found a packing of cost {cost} < closed UB {ub} '
           f'({rel}; check_solution {"ACCEPTS" if ok else "REJECTS"} it)')
    print(msg, flush=True)
    ledger.record(nm, variant, mid, 'FEASIBLE', ub=cost, time_s=dt, threads=workers, sol_file=rel, checked=int(ok),
                  note=f'second-method confirmation attempt at UB={ub}: packing of cost {cost} < UB '
                       f'-- REFUTES the closure if checked; {how}')
    with open(FLAG, 'a') as f:
        f.write(msg + '\n')
    return dict(outcome='REFUTED', status='FEASIBLE', ub_found=cost, sol_file=rel, checked=int(ok))


# ----------------------------------------------------------------------------- methods
def m_rep(nm, variant, ub, tl, workers, a):
    import sweep_rep
    r = sweep_rep.solve(nm, variant, tl, workers=workers, ub=ub - 1)
    st, dt = r['status'], r['time']
    if st == 'INFEASIBLE':
        return dict(outcome='CONFIRMED', status=st, time=dt,
                    detail=f'sweep_rep with obj<=UB-1={ub - 1} INFEASIBLE')
    if r['place'] is not None:
        rel, ok = store_and_check(nm, variant, MID['rep'], r['place'], r['val'])
        return dict(refuted(nm, variant, MID['rep'], ub, r['val'], rel, ok, dt, workers, 'sweep_rep'), time=dt)
    return dict(outcome='UNKNOWN', status=st, time=dt,
                detail=f'sweep_rep obj<=UB-1={ub - 1}: {st}; CP-SAT bound {r["lb_raw"]}')


def m_direct(nm, variant, ub, tl, workers, a):
    import cpsat_direct
    st, val, lb, dt, sol = cpsat_direct.solve(nm, tl, workers=workers, ub=ub - 1, area=(variant == 'area'),
                                             return_placement=True)
    if st == 'INFEASIBLE':
        return dict(outcome='CONFIRMED', status=st, time=dt,
                    detail=f'cpsat_direct with obj<=UB-1={ub - 1} INFEASIBLE')
    if val is not None:
        rel, ok = store_and_check(nm, variant, MID['direct'], sol['place'], val)
        return dict(refuted(nm, variant, MID['direct'], ub, val, rel, ok, dt, workers, f'cpsat_direct ({st})'),
                    time=dt)
    return dict(outcome='UNKNOWN', status=st, time=dt, detail=f'cpsat_direct obj<=UB-1={ub - 1}: {st}')


def m_lattice(nm, variant, ub, tl, workers, a):
    import lattice_exact as LX
    nv = lattice_size(nm, variant, ub)
    if nv > MAX_VECTORS:
        return dict(outcome='SKIPPED', status='TOO_MANY_VECTORS', time=0.0,
                    detail=f'> {MAX_VECTORS} vectors of cost <= UB-1')
    LX.METHOD = MID['lattice']   # file name of a packing it might find
    vec_tl = a.vec_time or max(5.0, tl / 3)
    short = min(5.0, vec_tl / 2)
    t0 = time.time()
    res = LX.run(nm, variant, tl, vec_tl, workers=workers, ub=ub - 1, record=False, short=short,
                 log=lambda s: print('   ', s, flush=True))
    dt = time.time() - t0
    assert res['ub_in'] <= ub - 1 and not res['ub_in_checked']
    s = res['stats']
    det = (f"lattice_exact vectors(cost<=UB-1)={s['enum']} cheap={s['cheap']} cpinf={s['cp_inf']} "
           f"unk={s['unknown']} reasons={res['reasons']} vec_tl={vec_tl:.0f} short={short:.0f}")
    if res['ub'] is not None:
        rel = ledger.sol_relpath(nm, variant, MID['lattice'], res['ub'])
        ok = recheck(nm, variant, rel, res['ub'])
        return dict(refuted(nm, variant, MID['lattice'], ub, res['ub'], rel, ok, dt, workers, 'lattice_exact'),
                    time=dt)
    if res['lb'] >= ub:
        return dict(outcome='CONFIRMED', status='LB_ONLY', time=dt,
                    detail=f'every bin-count vector of cost <= UB-1={ub - 1} refuted; {det}')
    return dict(outcome='UNKNOWN', status='LB_ONLY', time=dt,
                detail=f'refuted all vectors of cost < {res["lb"]} only; first_unknown={res["first_unknown"]}; {det}')


def m_lattice_indep(nm, variant, ub, tl, workers, a):
    nv = lattice_size(nm, variant, ub, indep=True)
    if nv > MAX_VECTORS:
        return dict(outcome='SKIPPED', status='TOO_MANY_VECTORS', time=0.0,
                    detail=f'> {MAX_VECTORS} raw vectors of cost <= UB-1')
    vec_tl = a.vec_time or max(5.0, tl / 3)
    env = dict(os.environ, NONEMPTY='1', SORTITEMS='1')
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(SRC, 'verify', 'verify_lattice_indep.py'), nm, variant,
                        str(ub), str(tl), str(vec_tl), str(workers)], capture_output=True, text=True, env=env,
                       cwd=ROOT)
    dt = time.time() - t0
    lines = p.stdout.strip().split('\n')
    for ln in lines[-6:]:
        print('   ', ln, flush=True)
    res = [ln for ln in lines if ln.startswith('RESULT')]
    if p.returncode != 0 or not res:
        return dict(outcome='ERROR', status='ERROR', time=dt, detail=(p.stderr or p.stdout)[-300:])
    r = res[-1]
    m = re.search(r'lb_proven=(\d+)', r)
    det = f'verify_lattice_indep NONEMPTY=1 SORTITEMS=1 target={ub} vec_tl={vec_tl:.0f}: {r}'
    if ' CONFIRMED ' in r and m and int(m.group(1)) >= ub:
        return dict(outcome='CONFIRMED', status='LB_ONLY', time=dt, detail=det)
    if ' REFUTED ' in r:
        # this script only reports that some vector of cost < UB is packable, without a packing
        msg = f'!!! REFUTATION CANDIDATE (no packing output): {nm} {variant}: {r}'
        print(msg, flush=True)
        with open(FLAG, 'a') as f:
            f.write(msg + '\n')
        return dict(outcome='REFUTED', status='FEASIBLE_VECTOR', time=dt, detail=det)
    return dict(outcome='UNKNOWN', status='LB_ONLY', time=dt, detail=det)


def m_cg(nm, variant, ub, tl, workers, a):
    out = os.path.join(OUT, f'bendex_{variant}.jsonl')
    cg_tl = min(30.0, max(10.0, tl / 3))
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(SRC, 'cg_bendex.py'), '--variant', variant, '--time', str(tl),
                        '--cg-time', str(cg_tl), '--check-time', str(a.cg_check_time), '--workers', str(workers),
                        '--out', out, nm],
                       capture_output=True, text=True, cwd=ROOT)
    dt = time.time() - t0
    lines = p.stdout.strip().split('\n')
    for ln in lines[-4:]:
        print('   ', ln[:400], flush=True)
    js = [json.loads(ln) for ln in lines if ln.startswith('{')]
    if p.returncode != 0 or not js:
        return dict(outcome='ERROR', status='ERROR', time=dt, detail=(p.stderr or p.stdout)[-300:])
    r = js[-1]
    det = (f"cg_bendex root={r['lb_root']} lb={r['lb']} cuts={r['cuts']} nogoods={r['nogoods']} "
           f"own_ub={r['own_ub']} cert={r['cert']} cg_time={cg_tl:.0f} check_time={a.cg_check_time:.0f}")
    if r['own_ub'] is not None and r['own_ub'] < ub:
        msg = f'!!! REFUTATION: {nm} {variant}: cg_bendex packing of cost {r["own_ub"]} < closed UB {ub}'
        print(msg, flush=True)
        with open(FLAG, 'a') as f:
            f.write(msg + '\n')
        return dict(outcome='REFUTED', status='FEASIBLE', time=dt, detail=det)
    if r['lb'] > ub or 'LB > checked UB' in p.stdout:
        print(f'!!! ANOMALY: {nm} {variant}: cg_bendex LB {r["lb"]} > checked UB {ub}', flush=True)
        return dict(outcome='ANOMALY_LB_ABOVE_UB', status='LB_ONLY', time=dt, detail=det)
    if r['lb'] >= ub:
        return dict(outcome='CONFIRMED', status='LB_ONLY', time=dt, detail=det)
    return dict(outcome='UNKNOWN', status='LB_ONLY', time=dt, detail=det)


METHODS = {'rep': m_rep, 'direct': m_direct, 'lattice': m_lattice, 'lattice_indep': m_lattice_indep, 'cg': m_cg}


def run_one(pr, meth, tl, workers, a):
    nm, variant, ub = pr['name'], pr['variant'], pr['ub']
    mid = MID[meth]
    print(f'== {nm} {variant} n={pr["n"]} UB={ub} fams={sorted(pr["fams"])} method={meth} tl={tl:.0f} '
          f'workers={workers}  [{time.strftime("%H:%M:%S")}]', flush=True)
    try:
        r = METHODS[meth](nm, variant, ub, tl, workers, a)
    except Exception as e:  # never record anything as proven on an exception
        r = dict(outcome='ERROR', status='ERROR', time=0.0, detail=repr(e)[:300])
    note_tail = r.get('detail', '')
    assert 'relies on' not in note_tail
    if meth != 'cg' and r['outcome'] == 'CONFIRMED':
        ledger.record(nm, variant, mid, 'LB_ONLY', lb=ub, time_s=r['time'], threads=workers,
                      note=f'second-method confirmation of closure at UB={ub} (independent of {pr["lb_methods"]}); '
                           f'self-contained; {note_tail}')
    elif meth != 'cg' and r['outcome'] in ('UNKNOWN', 'ERROR'):
        ledger.record(nm, variant, mid, 'UNKNOWN', time_s=r['time'], threads=workers,
                      note=f'second-method confirmation attempt at UB={ub}: no proof within {tl:.0f}s; {note_tail}')
    rec = dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'), name=nm, variant=variant, n=pr['n'], ub=ub,
               fams=sorted(pr['fams']), method=meth, method_id=mid, tl=tl, vec_tl=a.vec_time, workers=workers,
               **{k: v for k, v in r.items()})
    rec['time'] = round(float(rec.get('time') or 0.0), 1)
    with open(JSONL, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'   -> {r["outcome"]} ({r.get("status")}) in {rec["time"]}s: {note_tail[:300]}', flush=True)
    return r['outcome']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--time', type=float, default=60.0, help='time limit per (pair, method) run (s)')
    ap.add_argument('--vec-time', type=float, default=None, help='lattice per-vector limit (default time/3)')
    ap.add_argument('--cg-check-time', type=float, default=20.0, help='cg_bendex --check-time (exact fixed-count check)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--shard', default='0/1', help='i/k: this process takes pairs i, i+k, ...')
    ap.add_argument('--methods', default=None, help='comma list: only these methods (still in pair order)')
    ap.add_argument('--first-only', action='store_true', help='per pair, run only the first allowed untried method')
    ap.add_argument('--stop-at', default=None, help='local HH:MM; no run starts later, limits are cut to fit')
    ap.add_argument('--plan', action='store_true', help='print the pairs, families and method order, then exit')
    ap.add_argument('pairs', nargs='*', help='name:variant (default: all closed, unconfirmed pairs)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P = open_pairs()
    if a.pairs:
        sel = []
        for s in a.pairs:
            f = s.split(':')
            if len(f) == 4:   # explicit pair closed after results.csv was built: name:variant:UB:fam+fam
                sel.append(dict(name=f[0], variant=f[1], n=io.n_items(f[0]), ub=int(f[2]),
                                fams=set(f[3].split('+')), lb_methods='ledger rows after results.csv'))
            else:
                sel += [p for p in P if (p['name'], p['variant']) == tuple(f[:2])]
        P = sel
    else:  # cheap first: small n first, spread over shards
        P.sort(key=lambda p: (p['n'], p['variant'], p['name']))
    if a.plan:
        for p in P:
            print(p['name'], p['variant'], p['n'], p['ub'], '+'.join(sorted(p['fams'])), '->', ','.join(order_for(p['fams'])))
        print(len(P), 'pairs')
        return
    i0, k = map(int, a.shard.split('/'))
    P = P[i0::k]
    stop_ts = None
    if a.stop_at:
        hh, mm = map(int, a.stop_at.split(':'))
        lt = time.localtime()
        stop_ts = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        if stop_ts < time.time() - 6 * 3600:
            stop_ts += 86400
    only = set(a.methods.split(',')) if a.methods else None
    for pr in P:
        H = history()
        key = (pr['name'], pr['variant'])
        if any((h['name'], h['variant']) == key and h['outcome'] == 'CONFIRMED' for h in H):
            continue
        for meth in order_for(pr['fams']):
            if only is not None and meth not in only:
                continue
            if any((h['name'], h['variant'], h['method']) == key + (meth,) and h['tl'] >= a.time
                   and h['outcome'] in ('UNKNOWN', 'CONFIRMED', 'SKIPPED') for h in H):
                continue
            if os.path.exists(FLAG):
                print('REFUTATION_FOUND flag present: stopping', flush=True)
                return 3
            tl = a.time
            if stop_ts is not None:
                left = stop_ts - time.time()
                if left < 30:
                    print('stop-at reached', flush=True)
                    return 0
                tl = min(tl, left - 20)
            out = run_one(pr, meth, tl, a.workers, a)
            if out == 'REFUTED':
                return 3
            if out == 'CONFIRMED':
                break
            if a.first_only and out != 'SKIPPED':
                break
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
