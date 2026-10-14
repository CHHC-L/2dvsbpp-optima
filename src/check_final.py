"""Re-check the published table: every packing in results/final/ must pass src/check_solution.py at
exactly the UB of its row, and no LB may exceed its UB.

usage: python src/check_final.py
"""
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(r):
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'check_solution.py'),
                        os.path.join(ROOT, 'data', 'instances', r['name'] + '.txt'),
                        os.path.join(ROOT, r['solution']), r['variant']], capture_output=True, text=True)
    o = p.stdout.strip()
    ok = p.returncode == 0 and o.startswith('valid') and int(o.rsplit(' cost ', 1)[1]) == int(r['ub'])
    if r['lb'] and int(r['lb']) > int(r['ub']):
        ok = False
    return r, ok, o or p.stderr.strip()


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'final', 'results.csv'))))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, rows))
    bad = [(r, o) for r, ok, o in res if not ok]
    for r, o in bad:
        print('FAIL', r['name'], r['variant'], r['ub'], o)
    print(f'{len(rows) - len(bad)} / {len(rows)} packings re-checked at their table cost')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
