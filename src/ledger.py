"""Shared append-only results ledger for every solver run (2026-10-14 sprint).

Every script appends ONE row per (instance, variant, method) run to
results/ledger.csv via `record(...)`. Rows are short single writes with O_APPEND, so concurrent
appends from several processes are safe. Never edit or rewrite existing rows; a later, better
run simply adds a new row. `best()` folds the ledger into per-(instance, variant) best bounds.

Columns
  ts        local timestamp
  name      instance, e.g. MB_C3_11
  variant   'weighted' (file costs, P&S 2005) or 'area' (cost = W*H, heuristic literature / CSV)
  method    short id of the algorithm, e.g. cpsat_direct, cg_root, lattice_opp, packingsolver
  status    OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN / LB_ONLY / UB_ONLY ...
  lb        proven lower bound (integer, empty if none)
  ub        cost of a feasible packing found (integer, empty if none)
  time_s    wall time
  threads   CPU threads used
  sol_file  path (relative to project root) of the stored packing JSON for ub, if any;
            must be checkable with src/check_solution.py
  checked   1 if sol_file was validated by src/check_solution.py, else 0
  note      free text (no commas)
"""
import csv
import io as _io
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'results', 'ledger.csv')
COLS = ['ts', 'name', 'variant', 'method', 'status', 'lb', 'ub', 'time_s', 'threads',
        'sol_file', 'checked', 'note']


def record(name, variant, method, status, lb=None, ub=None, time_s=None, threads=None,
           sol_file='', checked=0, note=''):
    assert variant in ('weighted', 'area'), variant
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    row = [time.strftime('%Y-%m-%dT%H:%M:%S'), name, variant, method, status,
           '' if lb is None else int(lb), '' if ub is None else int(ub),
           '' if time_s is None else round(float(time_s), 1), '' if threads is None else threads,
           sol_file or '', int(checked), str(note).replace(',', ';').replace('\n', ' ')]
    buf = _io.StringIO()
    csv.writer(buf).writerow(row)
    line = buf.getvalue().encode()
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size == 0:
            hb = _io.StringIO(); csv.writer(hb).writerow(COLS)
            os.write(fd, hb.getvalue().encode())
        os.write(fd, line)
    finally:
        os.close(fd)


def sol_relpath(name, variant, method, cost, tag=''):
    """Path (relative to the project root) for a stored packing of the given cost.

    The cost is part of the file name, so a file is only ever rewritten with a packing of the same
    cost and a ledger row always points at a file of the cost it records. (Until 2026-10-14 several
    writers reused one name per (instance, variant, method) and later runs overwrote the evidence of
    older ledger rows; src/build_results.py re-checks every file before using it.)
    """
    tag = f'_{tag}' if tag else ''
    return os.path.join('results', 'solutions', 'sprint', f'{name}_{variant}_{method}{tag}_c{int(cost)}.json')


def rows():
    if not os.path.exists(LEDGER):
        return []
    return list(csv.DictReader(open(LEDGER)))


def best():
    """(name, variant) -> dict(lb, ub, lb_method, ub_method, ub_sol, ub_checked)."""
    out = {}
    for r in rows():
        k = (r['name'], r['variant'])
        d = out.setdefault(k, dict(lb=None, ub=None, lb_method='', ub_method='', ub_sol='',
                                   ub_checked=0))
        if r['lb'] not in ('', None):
            v = int(r['lb'])
            if d['lb'] is None or v > d['lb']:
                d['lb'], d['lb_method'] = v, r['method']
        # only CHECKED packings count as UBs (verify_primal found unchecked 'no improvement'
        # rows, re-costed from the other variant, undercutting the best checked UB)
        if r['ub'] not in ('', None) and int(r['checked'] or 0) == 1:
            v = int(r['ub'])
            if d['ub'] is None or v < d['ub']:
                d['ub'], d['ub_method'] = v, r['method']
                d['ub_sol'], d['ub_checked'] = r['sol_file'], int(r['checked'] or 0)
    return out
