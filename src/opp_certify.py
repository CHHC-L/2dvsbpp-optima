"""opp track: certify the per-bin infeasibility claims (no-goods) used by an opp_benders run.

For results/opp/benders/<name>_<variant>.json, every no-good (W, H, S) must be a set of items that
does not fit into a W x H bin.  Each is re-proved independently of CP-SAT:
  1. DFF witness (opp_core.dff_check; family members validated exhaustively), else
  2. SAT order encoding + CaDiCaL + drat-trim (VERIFIED), else 'UNCERTIFIED'.
The DFF rows of the master are DFF-valid by construction (exhaustive check).  The master itself
(assignment relaxation) is still decided by CP-SAT (not certified here).

usage: python src/opp_certify.py MB_C1_1_area [...]   (names of benders JSON files without .json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_core as oc  # noqa: E402
import opp_sat as osat  # noqa: E402

WORK = os.path.join('/private/tmp', 'opp_certify')


def certify(tag, tl=60):
    R = json.load(open(os.path.join(oc.ROOT, 'results', 'opp', 'benders', tag + '.json')))
    bins, items0 = oc.load(R['name'])
    order = sorted(range(len(items0)), key=lambda i: (-items0[i][0] * items0[i][1], i))
    items = [items0[i] for i in order]
    out = []
    for W, H, S in R.get('nogoods', []):
        its = [items[i] for i in S]
        w = oc.dff_check(its, W, H)
        if w is not None:
            out.append(dict(W=W, H=H, items=its, cert='dff', witness=w)); continue
        r = osat.solve(its, W, H, WORK, f'{tag}_{len(out)}', tl=tl)
        st = 'drat_verified' if r['status'] == 'UNSAT' and r.get('drat') == 'VERIFIED' else 'UNCERTIFIED_' + r['status']
        out.append(dict(W=W, H=H, items=its, cert=st))
        for f in (r['cnf'], r['proof']):
            if os.path.exists(f):
                os.remove(f)
    ok = all(o['cert'] in ('dff', 'drat_verified') for o in out)
    res = dict(tag=tag, n_nogoods=len(out), all_certified=ok, nogoods=out)
    json.dump(res, open(os.path.join(oc.ROOT, 'results', 'opp', 'certs', f'benders_{tag}.json'), 'w'))
    print(tag, 'nogoods', len(out), 'all certified', ok,
          {k: sum(1 for o in out if o['cert'] == k) for k in set(o['cert'] for o in out)}, flush=True)
    return res


if __name__ == '__main__':
    for t in sys.argv[1:]:
        certify(t)
