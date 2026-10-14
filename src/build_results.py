"""Build the final per-instance results table from the ledger (and the 2026-10-13 probe).

For every (instance, variant):
  * UB = cheapest ledger packing whose file still exists and that src/check_solution.py (run here
    again, as a subprocess) accepts with exactly the recorded cost. The file is copied to
    results/final/solutions/<name>_<variant>.json.
  * LB = largest proven lower bound over all ledger rows and the probe's CP-SAT runs.
  * Every LB source is mapped to the set of code bases ('families') it depends on (FAMILY below).
    A lattice_opp row that started from another row's LB ('relies on ledger LB X from M') depends
    on that row's families too. A closure is 'confirmed' if two LB sources that both reach the UB
    have disjoint family sets, i.e. no shared code.

Writes results/final/results.csv, results/final/SUMMARY.md and results/final/solutions/.

usage: .venv/bin/python src/build_results.py [--no-recheck]
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger  # noqa: E402
import vsbpp_io as io  # noqa: E402

ROOT = io.ROOT
OUT = os.path.join(ROOT, 'results', 'final')
PROBE = os.path.join(ROOT, 'results', 'probe-2026-10-13', 'all_runs.csv')

# method -> code base. Methods that share a model or a module are in the same family.
FAMILY = {
    'sweep_rep': 'rep',
    'cpsat_direct': 'direct', 'sweep_direct': 'direct', 'verify_cg_cpsat_direct': 'direct',
    'verify_opp_cpsat_direct': 'direct', 'verify_primal_cpsat_direct': 'direct',
    'verify_sweep_direct_lbm1': 'direct', 'probe_cpsat_direct': 'direct',
    'probe_cpsat_slots_v0': 'slots_v0',
    'cg_v2_root': 'cg', 'cg_v2_bendex': 'cg',
    'verify_cg_safe': 'cg',   # integer re-derivation of the cg LBs from the same duals (src/verify/cg_safe.py)
    'verify_cg_oldcg': 'cg_old',
    'lattice_opp': 'lattice', 'lattice_opp_fullonly': 'lattice',
    'verify_lattice_indep': 'lattice_indep',
    'opp_benders': 'opp', 'opp_cpsat': 'opp', 'opp_dff': 'opp', 'opp_stack1d': 'opp',
    'opp_master_highs': 'opp',
    'verify_opp_relax_highs': 'opp_indep', 'verify_opp_relax_cpsat': 'opp_indep',
    'verify_opp_parity': 'opp_indep', 'verify_opp_stack1d_highs': 'opp_indep',
    'verify_primal_areacover': 'area_bound',
    # second-method confirmation runs (src/verify/confirm_closures.py): same code as the method they call
    'confirm_rep': 'rep', 'confirm_direct': 'direct', 'confirm_lattice': 'lattice',
    'confirm_lattice_indep': 'lattice_indep',
}
RELIES = re.compile(r'relies on ledger LB (\d+) from (\S+)')
UNKNOWN_DEP = '?'   # a dependency that could not be resolved; never counts as independent


def lb_sources():
    """(name, variant) -> list of dict(lb, method, deps) in time order."""
    out = collections.defaultdict(list)
    if os.path.exists(PROBE):
        for r in csv.DictReader(open(PROBE)):
            if r['lb']:
                m = 'probe_' + r['model']
                out[(r['name'], r['variant'])].append(dict(lb=int(r['lb']), method=m, deps={FAMILY[m]}, ts=''))
    for r in sorted(ledger.rows(), key=lambda r: r['ts']):
        if r['lb'] in ('', None):
            continue
        m = r['method']
        if m not in FAMILY:
            raise SystemExit(f'unknown LB method {m!r}: add it to FAMILY')
        deps = {FAMILY[m]}
        mm = RELIES.search(r['note'])
        if mm:
            need, src = int(mm.group(1)), mm.group(2)
            prev = [s for s in out[(r['name'], r['variant'])] if s['method'] == src and s['lb'] >= need]
            deps |= min((s['deps'] for s in prev), key=len) if prev else {UNKNOWN_DEP}
        out[(r['name'], r['variant'])].append(dict(lb=int(r['lb']), method=m, deps=deps, ts=r['ts']))
    return out


def check(nm, variant, rel):
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'check_solution.py'), io.instance_path(nm),
                        os.path.join(ROOT, rel), variant], capture_output=True, text=True)
    o = p.stdout.strip()
    return int(o.rsplit(' cost ', 1)[1]) if p.returncode == 0 and o.startswith('valid') else None


def ub_candidates():
    """(name, variant) -> checked ledger rows with a packing file, cheapest first."""
    c = collections.defaultdict(list)
    for r in ledger.rows():
        if r['ub'] not in ('', None) and str(r['checked']) == '1' and r['sol_file']:
            c[(r['name'], r['variant'])].append((int(r['ub']), r['method'], r['sol_file']))
    for k in c:
        c[k] = sorted(set(c[k]))
    return c


def best_ub(key, cands, recheck):
    """cheapest candidate whose file exists and re-checks at the recorded cost; also the number of
    strictly cheaper candidates whose evidence is gone (file overwritten or missing)."""
    lost = []
    for ub, m, rel in cands:
        if not os.path.exists(os.path.join(ROOT, rel)):
            lost.append(ub)
            continue
        if recheck:
            if check(key[0], key[1], rel) != ub:
                lost.append(ub)
                continue
        else:
            try:
                if json.load(open(os.path.join(ROOT, rel))).get('cost', ub) != ub:
                    lost.append(ub)
                    continue
            except Exception:
                lost.append(ub)
                continue
        return ub, m, rel, sum(1 for x in lost if x < ub)
    return None, '', '', len(lost)


def confirmed(srcs, ub):
    """True if two sources with LB >= ub have disjoint dependency sets (none unresolved)."""
    ok = [s['deps'] for s in srcs if s['lb'] >= ub and UNKNOWN_DEP not in s['deps']]
    return any(not (a & b) for i, a in enumerate(ok) for b in ok[i + 1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--no-recheck', action='store_true', help='trust stored costs instead of re-running the checker')
    a = ap.parse_args()
    if not ledger.rows():
        raise SystemExit('no run ledger at results/ledger.csv: nothing to build from '
                         '(to re-check the published table use src/check_final.py)')
    srcs = lb_sources()
    cands = ub_candidates()
    keys = [(nm, v) for v in ('weighted', 'area') for nm in io.all_names()]
    with ThreadPoolExecutor(8) as ex:
        ubs = dict(zip(keys, ex.map(lambda k: best_ub(k, cands.get(k, []), not a.no_recheck), keys)))

    os.makedirs(os.path.join(OUT, 'solutions'), exist_ok=True)
    rows = []
    for nm, v in keys:
        ub, ub_m, rel, lost = ubs[(nm, v)]
        S = srcs.get((nm, v), [])
        lb = max((s['lb'] for s in S), default=None)
        if lb is not None and ub is not None and lb > ub:
            raise SystemExit(f'CONTRADICTION {nm} {v}: LB {lb} > checked UB {ub}')
        closed = lb is not None and lb == ub
        at_lb = sorted({s['method'] for s in S if s['lb'] == lb}) if lb is not None else []
        fams = sorted(set().union(*[s['deps'] for s in S if lb is not None and s['lb'] == lb])) if lb is not None else []
        bks, bnd = io.bk[nm]
        sol = ''
        if rel:
            sol = os.path.join('results', 'final', 'solutions', f'{nm}_{v}.json')
            shutil.copyfile(os.path.join(ROOT, rel), os.path.join(ROOT, sol))
        rows.append(dict(
            name=nm, cls=int(nm.split('_')[1][1:]), n=io.n_items(nm), variant=v, lb=lb, ub=ub,
            gap_pct=round(100 * (ub - lb) / ub, 3) if (lb is not None and ub) else '',
            closed=int(closed), confirmed=int(closed and confirmed(S, ub)),
            lb_methods=' '.join(at_lb), lb_families=' '.join(fams), ub_method=ub_m, ub_source=rel,
            lost_cheaper_evidence=lost, solution=sol,
            csv_bks=bks if v == 'area' else '', csv_bound=bnd if v == 'area' else '',
            csv_closed=int(bks == bnd) if v == 'area' else ''))
    with open(os.path.join(OUT, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_summary(rows)


def write_summary(rows):
    ns = (20, 40, 60, 80, 100)
    L = ['# Final results (generated by `src/build_results.py`; do not edit by hand)', '',
         'Closed = our best proven LB equals the cost of a checked packing. Confirmed = the LB is reached',
         'by two sources that share no code (see FAMILY in `src/build_results.py`).', '']
    for v, title in (('weighted', 'W (file costs, the Pisinger–Sigurd problem)'),
                     ('area', 'U (cost = bin area, the heuristic literature and the packingsolver CSV)')):
        R = [r for r in rows if r['variant'] == v]
        L += [f'## {title}', '', '| | ' + ' | '.join(f'n={n}' for n in ns) + ' | total |',
              '|---|' + '---|' * (len(ns) + 1)]

        def line(label, pred):
            c = [sum(1 for r in R if r['n'] == n and pred(r)) for n in ns]
            return f'| {label} | ' + ' | '.join(map(str, c)) + f' | **{sum(c)}** |'
        L.append(line('closed', lambda r: r['closed']))
        L.append(line('closed and confirmed by 2 independent methods', lambda r: r['confirmed']))
        L.append(line('closed, 1 method only', lambda r: r['closed'] and not r['confirmed']))
        L.append(line('no LB of ours', lambda r: r['lb'] is None))
        L.append(line('no checked packing', lambda r: r['ub'] is None))
        if v == 'area':
            L.append(line('closed in the CSV (Wei et al. 2013 bound = BKS)', lambda r: r['csv_closed']))
            L.append(line('closed by us, not in the CSV', lambda r: r['closed'] and not r['csv_closed']))
            L.append(line('our UB < CSV BKS', lambda r: r['ub'] is not None and r['ub'] < r['csv_bks']))
            L.append(line('our UB = CSV BKS', lambda r: r['ub'] is not None and r['ub'] == r['csv_bks']))
            L.append(line('our UB > CSV BKS', lambda r: r['ub'] is not None and r['ub'] > r['csv_bks']))
            L.append(line('our LB > CSV bound', lambda r: r['lb'] is not None and r['lb'] > r['csv_bound']))
        L.append('')
        withlb = [r for r in R if r['lb'] is not None and r['ub'] is not None]
        L.append(f'Mean gap (UB − LB)/UB over the {len(withlb)} instances with an LB of ours: '
                 f'{sum(r["gap_pct"] for r in withlb) / max(1, len(withlb)):.2f} %.')
        L.append('')
    better = [r for r in rows if r['variant'] == 'area' and r['ub'] is not None and r['ub'] < r['csv_bks']]
    if better:
        L += ['## Area variant: packings cheaper than the 2025 CSV BKS', '',
              '| Instance | ours | CSV BKS | proven optimal |', '|---|---|---|---|']
        L += [f'| {r["name"]} | {r["ub"]} | {r["csv_bks"]} | {"yes" if r["closed"] else "no"} |' for r in better]
        L.append('')
    lost = [r for r in rows if r['lost_cheaper_evidence']]
    if lost:
        L += [f'{len(lost)} pairs had a cheaper ledger UB row whose packing file was later overwritten; '
              'the table uses the cheapest packing that still re-checks.', '']
    open(os.path.join(OUT, 'SUMMARY.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
