"""opp track: randomised self-test of the OPP deciders (catches encoding / symmetry bugs).

For random small instances (with deliberate duplicate items, near-full area) compare
  A  CP-SAT plain (no normal patterns, no symmetry breaking)   -- reference
  B  CP-SAT with normal patterns + reflection + identical-item order
  C  SAT order encoding with all reductions (CaDiCaL; UNSAT proofs checked by drat-trim)
  D  SAT order encoding plain
  E  DFF test (may only say 'infeasible'; must never contradict a feasible answer)
Any disagreement is printed as MISMATCH.  SAT answers are geometry-verified.

usage: python src/verify/opp_selftest.py [n_instances] [seed]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opp_core as oc  # noqa: E402
import opp_cpsat as ocp  # noqa: E402
import opp_sat as osat  # noqa: E402

WORK = os.path.join('/private/tmp', 'opp_selftest')


def rand_inst(rng):
    W = rng.randint(4, 10); H = rng.randint(4, 10)
    items = []
    target = W * H * rng.uniform(0.75, 1.0)
    area = 0
    while area < target and len(items) < 9:
        if items and rng.random() < 0.3:
            it = rng.choice(items)
        else:
            it = (rng.randint(1, max(1, W * 2 // 3)), rng.randint(1, max(1, H * 2 // 3)))
        if area + it[0] * it[1] > W * H:
            break
        items.append(it); area += it[0] * it[1]
    return items, W, H


def main(N=150, seed=1):
    rng = random.Random(seed)
    stats = {'feas': 0, 'infeas': 0, 'mismatch': 0, 'drat_ok': 0, 'dff_inf': 0}
    for k in range(N):
        items, W, H = rand_inst(rng)
        a = ocp.opp_single(items, W, H, tl=30, sym=False, normal=False)[0]
        b = ocp.opp_single(items, W, H, tl=30, sym=True, normal=True)[0]
        c = osat.solve(items, W, H, WORK, f'st{k}c', tl=30)
        d = osat.solve(items, W, H, WORK, f'st{k}d', tl=30, normal=False, reflect=False, ident=False)
        e = oc.dff_check(items, W, H)
        cs = {'SAT': 'FEASIBLE', 'UNSAT': 'INFEASIBLE'}.get(c['status'], c['status'])
        ds = {'SAT': 'FEASIBLE', 'UNSAT': 'INFEASIBLE'}.get(d['status'], d['status'])
        ok = a == b == cs == ds and (e is None or a == 'INFEASIBLE')
        for r in (c, d):
            if r['status'] == 'UNSAT':
                if r.get('drat') == 'VERIFIED':
                    stats['drat_ok'] += 1
                else:
                    ok = False
        if e is not None:
            stats['dff_inf'] += 1
        stats['feas' if a == 'FEASIBLE' else 'infeas'] += 1
        if not ok:
            stats['mismatch'] += 1
            print('MISMATCH', k, W, H, items, a, b, cs, ds, e, c.get('drat'), d.get('drat'), flush=True)
        for f in os.listdir(WORK):
            os.remove(os.path.join(WORK, f))
    print('selftest', N, 'seed', seed, stats, flush=True)
    return stats


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
