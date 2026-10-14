"""End-to-end smoke test of the solver drivers on a throw-away copy of the project.

Every driver must (1) write packings under ledger.sol_relpath names (cost in the file name),
(2) record ledger rows whose sol_file re-checks at exactly the recorded UB, and (3) never record an
LB above a checked UB. Then src/build_results.py must run on the resulting ledger.
Takes about 2 minutes. usage (from the project root):
    .venv/bin/python -m unittest tests.test_pipeline
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [  # (driver, args) -- small instances, short limits
    ('sweep_run.py', ['--variant', 'weighted', '--time', '10', '--workers', '2', 'MB_C1_1']),
    ('cg_v2.py', ['--variant', 'area', '--time', '20', '--workers', '2', '--out', 'results/cg.jsonl', 'MB_C9_1']),
    ('cg_bendex.py', ['--variant', 'weighted', '--time', '20', '--cg-time', '10', '--workers', '2',
                      '--out', 'results/bendex.jsonl', 'MB_C9_2']),
    ('lattice_exact.py', ['--variant', 'weighted', '--time', '30', '--workers', '2', 'MB_C2_1']),
    ('opp_benders.py', ['--variant', 'area', '--time', '30', '--record', 'MB_C2_2']),
    ('primal_heur.py', ['--variant', 'area', '--tl', '1', '--names', 'MB_C3_1']),
    ('primal_lns.py', ['--variant', 'area', '--tl', '5', '--names', 'MB_C3_1']),
]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix='vsbpp_smoke_')
        shutil.copytree(os.path.join(ROOT, 'src'), os.path.join(cls.d, 'src'),
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(os.path.join(ROOT, 'data'), os.path.join(cls.d, 'data'))
        os.makedirs(os.path.join(cls.d, 'results', 'solutions', 'sprint'))
        os.makedirs(os.path.join(cls.d, 'results', 'opp', 'benders'))
        cls.logs = {}
        for drv, args in RUNS:
            r = subprocess.run([sys.executable, os.path.join('src', drv)] + args, cwd=cls.d,
                               capture_output=True, text=True, timeout=600)
            cls.logs[drv] = (r.returncode, r.stdout[-2000:] + r.stderr[-2000:])
        with open(os.path.join(cls.d, 'results', 'ledger.csv')) as f:
            cls.rows = list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_drivers_exit_cleanly(self):
        for drv, (rc, log) in self.logs.items():
            self.assertEqual(rc, 0, f'{drv} failed:\n{log}')

    def test_every_driver_recorded(self):
        meth = {r['method'] for r in self.rows}
        for m in ('sweep_rep', 'cg_v2_root', 'cg_v2_bendex', 'lattice_opp', 'opp_benders', 'primal_heur', 'primal_lns'):
            self.assertIn(m, meth)

    def test_solution_files(self):
        n = 0
        for r in self.rows:
            if r['sol_file']:
                n += 1
                self.assertRegex(r['sol_file'], r'_c%s\.json$' % r['ub'], r)
                self.assertEqual(r['checked'], '1', r)
                out = subprocess.run([sys.executable, 'src/check_solution.py', f'data/instances/{r["name"]}.txt',
                                      r['sol_file'], r['variant']], cwd=self.d, capture_output=True, text=True).stdout
                self.assertTrue(out.strip().endswith(f'cost {r["ub"]}'), (r, out))
            elif r['ub']:
                self.assertEqual(r['checked'], '0', r)
        self.assertGreater(n, 5)

    def test_no_lb_above_checked_ub(self):
        best = {}
        for r in self.rows:
            if r['ub'] and r['checked'] == '1':
                k = (r['name'], r['variant']); best[k] = min(best.get(k, 10 ** 12), int(r['ub']))
        for r in self.rows:
            if r['lb']:
                self.assertLessEqual(int(r['lb']), best.get((r['name'], r['variant']), 10 ** 12), r)

    def test_build_results(self):
        r = subprocess.run([sys.executable, 'src/build_results.py'], cwd=self.d, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(re.search(r'\| closed \|', r.stdout))


if __name__ == '__main__':
    unittest.main()
