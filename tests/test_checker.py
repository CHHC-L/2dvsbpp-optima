"""Unit tests for the standalone checker src/check_solution.py.

usage (from the project root): .venv/bin/python -m unittest discover tests
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, 'src', 'check_solution.py')

# 2 bin types (10x10 cost 7, 6x4 cost 3), 3 items 6x4, 4x10, 6x6
INSTANCE = '3 2 1\n10 10 1 7\n6 4 1 3\n1 6 4 1\n2 4 10 1\n3 6 6 1\n'
GOOD = [[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 0], [1, 0, 6, 4, 0, 0]]


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.inst = os.path.join(self.d, 'i.txt')
        open(self.inst, 'w').write(INSTANCE)

    def run_check(self, place, variant='weighted'):
        f = os.path.join(self.d, 's.json')
        json.dump({'place': place}, open(f, 'w'))
        r = subprocess.run([sys.executable, CHECK, self.inst, f, variant], capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    def test_valid(self):
        self.assertEqual(self.run_check(GOOD), (0, 'valid; bins used [0, 1] cost 10'))
        self.assertEqual(self.run_check(GOOD, 'area')[1], 'valid; bins used [0, 1] cost 124')

    def assertRejected(self, place):
        self.assertNotEqual(self.run_check(place)[0], 0, place)

    def test_overlap(self):
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 3, 0], [1, 0, 6, 4, 0, 0]])

    def test_outside(self):
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 5], [1, 0, 6, 4, 0, 0]])

    def test_rotated_item(self):
        self.assertRejected([[0, 0, 10, 4, 0, 0], [0, 0, 6, 6, 0, 4], [1, 0, 6, 4, 0, 0]])

    def test_missing_and_duplicate_item(self):
        self.assertRejected(GOOD[:2])
        self.assertRejected(GOOD + [GOOD[2][:4] + [0, 0]])

    def test_negative_bin_type(self):  # Python's bins[-1] would silently pick the last bin line
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 0], [-1, 0, 6, 4, 0, 0]])

    def test_bin_type_out_of_range(self):
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 0], [2, 0, 6, 4, 0, 0]])

    def test_non_integer_and_negative_coordinates(self):
        self.assertRejected([[0, 0, 4, 10, 0.5, 0], [0, 0, 6, 6, 4, 0], [1, 0, 6, 4, 0, 0]])
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 0], [1, -1, 6, 4, 0, 0]])
        self.assertRejected([[0, 0, 4, 10, 0, 0], [0, 0, 6, 6, 4, 0], [1, 0, 6, 4, 0, -1]])

    def test_separate_copies_do_not_overlap(self):  # same type, different copy index k
        self.assertEqual(self.run_check([[0, 0, 4, 10, 0, 0], [0, 1, 6, 6, 0, 0], [1, 0, 6, 4, 0, 0]])[1],
                         'valid; bins used [0, 0, 1] cost 17')


if __name__ == '__main__':
    unittest.main()
