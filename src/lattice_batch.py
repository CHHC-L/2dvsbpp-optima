"""Batch driver for src/lattice_exact.py over (instance, variant) pairs (lattice track).

usage: python src/lattice_batch.py <time_per_pair_s> <workers> <stream_id> <n_streams> <deadline_epoch> pair ...
  env LATTICE_SHORT (default 5 s; 0 = single pass) = pass-1 per-vector limit.
  pair = MB_C8_11:area ; pairs are split round-robin over streams; a pair already closed in the
  ledger (best lb >= best checked ub) is skipped; nothing new is started after deadline_epoch, and
  each run's time limit is capped so it ends before it.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ledger  # noqa: E402
import lattice_exact  # noqa: E402

tl, workers, sid, ns, deadline = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
    float(sys.argv[5])
pairs = sys.argv[6:][sid::ns]
SHORT = float(os.environ.get('LATTICE_SHORT', '5')) or None  # pass-1 per-vector limit (two-pass mode)
BFRAC = float(os.environ.get('LATTICE_BENDERS_FRAC', '0.5'))  # share of each vector's limit for the Benders loop
USE_L = os.environ.get('LATTICE_USE_LEDGER_LB', '0') == '1'  # skip vectors below the ledger LB (LB then relies on it)
for p in pairs:
    nm, v = p.split(':')
    rem = deadline - time.time()
    if rem < 20:
        print('# deadline reached', flush=True)
        break
    b = ledger.best().get((nm, v), {})
    if b.get('lb') is not None and b.get('ub') is not None and b.get('ub_checked') and b['lb'] >= b['ub']:
        print(f'# skip {nm} {v}: closed in ledger {b}', flush=True)
        continue
    t = min(tl, rem - 5)
    try:
        lattice_exact.run(nm, v, t, t, workers=workers, short=SHORT, use_ledger_lb=USE_L, benders_frac=BFRAC, log=lambda s: print(s, flush=True))
    except Exception as e:  # keep the batch alive
        print(f'# ERROR {nm} {v}: {e!r}', flush=True)
