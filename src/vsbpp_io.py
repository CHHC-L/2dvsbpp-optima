"""Instance / BKS I/O for the Pisinger-Sigurd 2DVSBPP benchmark (500 'MB' instances).

Instance file format (original Pisinger-Sigurd files, data/instances/MB_C<c>_<j>.txt):
    line 0:            n  m  1                 (n items, m bin-type lines)
    lines 1..m:        W  H  1  cost           (one bin type per line; supply is unlimited)
    lines m+1..m+n:    idx  w  h  1            (one item per line, demand 1, NO rotation)

Two cost variants are used on the same files:
    'weighted' : cost = 4th column of the bin line (the Pisinger & Sigurd 2005 problem)
    'area'     : cost = W*H (the 'unweighted' variant of Wei et al. 2013 and of the packingsolver
                 BKS csv)
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST_DIR = os.path.join(ROOT, 'data', 'instances')
BKS_CSV = os.path.join(ROOT, 'data', 'bks_packingsolver_2025-03.csv')

TRANSPOSE = False  # set True only to test the (wrong) transposed reading convention


def instance_path(nm):
    return os.path.join(INST_DIR, nm + '.txt')


def all_names():
    return [f'MB_C{c}_{j}' for c in range(1, 11) for j in range(1, 51)]


def n_items(nm):
    """n = 20, 40, 60, 80, 100 for j in 1-10, 11-20, 21-30, 31-40, 41-50."""
    j = int(nm.split('_')[2])
    return 20 * ((j - 1) // 10 + 1)


def read(nm, area=False):
    """Return (bins, items): bins = [(W, H, cost)], items = [(w, h)]."""
    L = open(instance_path(nm)).read().split('\n')
    n, m, _ = map(int, L[0].split())
    bins = [tuple(map(int, L[1 + t].split())) for t in range(m)]
    items = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
    if TRANSPOSE:
        items = [(h, w) for (w, h) in items]
    return [(b[0], b[1], (b[0] * b[1] if area else b[3])) for b in bins], items


def load_bks():
    """packingsolver csv (area variant!): name -> (best known solution value, best known bound)."""
    bk = {}
    for r in csv.DictReader(open(BKS_CSV)):
        bk[r['Path'].split('/')[-1]] = (int(r['Best known solution value']), int(r['Best known bound']))
    return bk


bk = load_bks()
