"""verify_cg: try to REFUTE claimed LBs with the probe's direct CP-SAT model (src/cpsat_direct.py, other code):
solve with obj <= LB-1, 1 worker, short limit.  INFEASIBLE => independent proof OPT >= LB;
FEASIBLE => packing cheaper than the claimed LB (refutation, saved + checked); UNKNOWN => nothing."""
import json, os, sys, time, subprocess
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SRC); sys.path.insert(0, SRC); os.chdir(ROOT)
import cpsat_direct as D
import ledger
tl = float(sys.argv[1]); stop = sys.argv[2]   # stop 'HH:MM'
todo = [a.split(':') for a in sys.argv[3:]]    # name:variant:LB
hh, mm = map(int, stop.split(':'))
lt = time.localtime(); stop_ts = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
for nm, var, L in todo:
    L = int(L); left = stop_ts - time.time()
    if left < 10: print('stop reached', flush=True); break
    t = min(tl, left - 5)
    st, val, lb, dt, P = D.solve(nm, t, workers=1, ub=L - 1, area=(var == 'area'), return_placement=True)
    print(json.dumps(dict(name=nm, variant=var, L=L, status=st, val=val, t=round(dt, 1))), flush=True)
    if st == 'INFEASIBLE':
        ledger.record(nm, var, 'verify_cg_cpsat_direct', 'INFEASIBLE', lb=L, time_s=dt, threads=1,
                      note=f'cpsat_direct with obj<=L-1={L-1} INFEASIBLE: independent confirmation of cg LB {L}')
    elif st.startswith('OPTIMAL') or st.startswith('FEASIBLE'):
        f = f'results/verify_cg/{nm}_{var}_refute.json'
        json.dump(dict(place=[list(p) for p in P['place']]), open(f, 'w'))
        out = subprocess.run([sys.executable, 'src/check_solution.py', f'data/instances/{nm}.txt', f, var], capture_output=True, text=True).stdout
        print('!!! REFUTATION CANDIDATE', nm, var, val, '<', L, out, flush=True)
        ledger.record(nm, var, 'verify_cg_cpsat_direct', 'FEASIBLE', ub=val, time_s=dt, threads=1, sol_file=f,
                      checked=int(out.startswith('valid') and int(out.split('cost')[-1]) == val),
                      note=f'packing below claimed cg LB {L} -- REFUTES it if checked')
    else:
        ledger.record(nm, var, 'verify_cg_cpsat_direct', 'UNKNOWN', time_s=dt, threads=1,
                      note=f'obj<=LB-1={L-1}: no answer in {t:.0f}s (neither confirms nor refutes cg LB {L})')
