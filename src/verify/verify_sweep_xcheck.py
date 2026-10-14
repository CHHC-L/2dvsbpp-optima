"""verify_sweep (sprint 2026-10-14): independent LB cross-check with a DIFFERENT formulation.

For each (name, variant, LB) in the queue, solve the ORIGINAL probe model src/cpsat_direct.py
(per-type slots, no dominance, no subset-area cuts, no lattice rounding, no hints, shares no model
code with sweep_rep.py) with obj <= LB - 1:
  INFEASIBLE        -> no packing of cost <= LB-1 exists -> LB independently confirmed
  FEASIBLE/OPTIMAL  -> a packing cheaper than the claimed LB: LB REFUTED (packing stored + checked)
  UNKNOWN           -> unconfirmed (a time-out never counts as infeasible)
usage: .venv/bin/python src/verify/verify_sweep_xcheck.py <queue.txt> <shard i/k> <tl> <workers> <deadline_epoch>
queue line: name variant LB
"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vsbpp_io as io
import ledger
import cpsat_direct
ROOT = io.ROOT
OUT = os.path.join(ROOT, 'results', 'verify_sweep')
qf, shard, tl, workers, deadline = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
i0, k = map(int, shard.split('/'))
Q = [l.split() for l in open(qf) if l.strip() and not l.startswith('#')][i0::k]
logf = os.path.join(OUT, f'xcheck_{os.path.basename(qf)}_{i0}of{k}.log')
for nm, variant, lb in Q:
    lb = int(lb)
    if time.time() + tl + 10 > deadline:
        print('deadline; stop before', nm, flush=True); break
    st, v, blb, dt, sol = cpsat_direct.solve(nm, tl, workers=workers, ub=lb - 1, area=(variant == 'area'),
                                             return_placement=True)
    verdict = 'unconfirmed'; rel = ''; checked = 0; note = f'cpsat_direct with obj<=LB-1={lb-1}; claimed LB {lb}'
    if st == 'INFEASIBLE':
        verdict = 'confirmed'
        ledger.record(nm, variant, 'verify_sweep_direct_lbm1', 'LB_ONLY', lb=lb, time_s=dt, threads=workers,
                      note=note + ' INFEASIBLE -> LB confirmed by independent formulation')
    elif v is not None:
        verdict = 'REFUTED'
        rel = f'results/solutions/sprint/{nm}_{variant}_verify_sweep_refute.json'
        json.dump(dict(name=nm, variant=variant, place=[list(map(int, p)) for p in sol['place']], cost=v),
                  open(os.path.join(ROOT, rel), 'w'))
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'check_solution.py'), io.instance_path(nm),
                            os.path.join(ROOT, rel), variant], capture_output=True, text=True, cwd=ROOT)
        checked = int(r.stdout.startswith('valid') and r.stdout.strip().endswith(f'cost {v}'))
        ledger.record(nm, variant, 'verify_sweep_direct_lbm1', 'FEASIBLE', ub=v, time_s=dt, threads=workers,
                      sol_file=rel, checked=checked, note=note + f' FOUND packing {v} < claimed LB; check={r.stdout.strip()}')
    else:
        ledger.record(nm, variant, 'verify_sweep_direct_lbm1', 'UNKNOWN', time_s=dt, threads=workers,
                      note=note + ' time-out; no conclusion')
    line = f'{time.strftime("%H:%M:%S")} {nm} {variant} LB={lb} status={st} val={v} t={dt:.1f} verdict={verdict} checked={checked}'
    print(line, flush=True)
    open(logf, 'a').write(line + '\n')
