"""opp track: batch driver for opp_benders over many instances (1 thread per process).

For each (name, variant): U = min(CSV area BKS, best CHECKED ledger UB for that variant);
run Benders with cost <= U (i.e. ub = U + 1) so that an optimal packing is produced and checked.
Results go to the ledger (method opp_benders) and results/opp/benders/<name>_<variant>.json.

usage: python src/opp_batch.py --variants area weighted --time 60 --deadline 01:05 NAME [NAME ...]
       NAME may be a range spec like 'n20' (all j=1..10) or 'n40' (j=11..20).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402
import opp_benders as ob  # noqa: E402
import ledger  # noqa: E402


def names_of(spec):
    if spec == 'n20':
        return [f'MB_C{c}_{j}' for j in range(1, 11) for c in range(1, 11)]
    if spec == 'n40':
        return [f'MB_C{c}_{j}' for j in range(11, 21) for c in range(1, 11)]
    return [spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['area', 'weighted'])
    ap.add_argument('--time', type=float, default=60.0)
    ap.add_argument('--master-time', type=float, default=40.0)
    ap.add_argument('--opp-time', type=float, default=10.0)
    ap.add_argument('--deadline', default=None, help='HH:MM local; no new run starts after it')
    ap.add_argument('--skip-closed', action='store_true', help='skip (name,variant) already closed in ledger')
    ap.add_argument('specs', nargs='+')
    A = ap.parse_args()
    names = [n for s in A.specs for n in names_of(s)]
    dl = None
    if A.deadline:
        hh, mm = map(int, A.deadline.split(':'))
        now = datetime.datetime.now()
        dl = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if dl < now:
            dl += datetime.timedelta(days=1)
    for nm in names:
        for var in A.variants:
            if dl is not None:
                left = (dl - datetime.datetime.now()).total_seconds()
                if left < 5:
                    print('deadline reached', flush=True); return
            else:
                left = 1e9
            best = ledger.best().get((nm, var))
            if A.skip_closed and best and best['lb'] is not None and best['ub'] is not None \
                    and best['lb'] >= best['ub'] and best['ub_checked']:
                print(nm, var, 'already closed', best['lb'], flush=True)
                continue
            U = ob._ub_default(nm, var)
            tl = min(A.time, left)
            B = ob.Benders(nm, var, U + 1, opp_tl=A.opp_time, log=lambda s: None)
            r = B.run(tl, min(A.master_time, tl))
            sol_file = ''; checked = 0
            if r.get('place'):
                sol_file = ledger.sol_relpath(nm, var, 'opp_benders', r['ub'])
                json.dump(dict(name=nm, variant=var, place=r['place']), open(os.path.join(oc.ROOT, sol_file), 'w'))
                out = subprocess.run([sys.executable, os.path.join(oc.ROOT, 'src', 'check_solution.py'),
                                      os.path.join(oc.ROOT, 'data', 'instances', nm + '.txt'),
                                      os.path.join(oc.ROOT, sol_file), var], capture_output=True, text=True).stdout.strip()
                checked = int(out.startswith('valid') and out.endswith(f'cost {r["ub"]}'))
            lb = r.get('lb')
            status = r['status']
            if status == 'OPTIMAL_UB':      # nothing with cost <= U: LB = U + 1 (U itself unverified here)
                lb = U + 1
            if status == 'OPTIMAL' and not checked:
                status = 'OPTIMAL_UNCHECKED'
            ledger.record(nm, var, 'opp_benders', status, lb=lb, ub=r.get('ub') if checked else None,
                          time_s=r['time'], threads=1, sol_file=sol_file if checked else '', checked=checked,
                          note=f'logic Benders (CP-SAT master+DFF rows+OPP nogoods); cost<=U={U}; iters {r["iters"]}; '
                               f'nogoods {len(r["nogoods"])}; dff rows {r["dff_rows"]}; opp calls {r["opp_calls"]}; tl {tl:.0f}')
            json.dump(dict(name=nm, variant=var, U=U, **{k: v for k, v in r.items() if k != 'place'}),
                      open(os.path.join(oc.ROOT, 'results', 'opp', 'benders', f'{nm}_{var}.json'), 'w'))
            print(nm, var, 'U', U, status, 'lb', lb, 'ub', r.get('ub'), 'checked', checked,
                  'it', r['iters'], 't', round(r['time'], 1), time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
