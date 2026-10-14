"""Sweep track driver (sprint 2026-10-14): resumable CP-SAT sweep over (instance, variant) pairs.

For every pair, in the given order:
  1. re-read ledger.best(); skip the pair if it is closed (best lb == best ub, ub packing checked);
  2. collect every known packing of the instance (ledger rows with checked=1 in EITHER variant,
     the probe file results/solutions/<name>_area.json, our own sweep files), re-verify each one
     geometrically in-process and re-cost it in the target variant; the cheapest one becomes the
     solution hint and UB = min(CSV area BKS, its cost)  (the area BKS is >= OPT in both variants);
  3. solve src/sweep_model.solve under the time limit (early stop if a solution reaches the best
     LB known in the ledger; that LB is never put into the model);
  4. store the packing at results/solutions/sprint/<name>_<variant>_<method>_c<cost>.json
     (ledger.sol_relpath), run src/check_solution.py on it, and record one ledger row. If the
     packing is also cheaper in the other variant than that variant's best ledger UB, store/record
     it under method sweep_cross too.

usage (from the project root):
  .venv/bin/python src/sweep_run.py --variant weighted --time 20 --workers 3 --group 20 [--model rep|direct]
  .venv/bin/python src/sweep_run.py --variant area --time 40 --workers 1 --group 40 --shard 0/3
  .venv/bin/python src/sweep_run.py --variant weighted --time 40 MB_C1_11 MB_C2_11
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsbpp_io as io  # noqa: E402
import ledger  # noqa: E402

ROOT = io.ROOT
SOL_DIR = os.path.join(ROOT, 'results', 'solutions', 'sprint')
OUT_DIR = os.path.join(ROOT, 'results', 'sweep')
PY = sys.executable
METHOD = 'sweep_rep'          # set from --model: sweep_rep (src/sweep_rep.py) or sweep_direct (src/sweep_model.py)
SOL_METHODS = ('sweep_rep', 'sweep_direct', 'sweep_cross')


def other(variant):
    return 'area' if variant == 'weighted' else 'weighted'


def verify_place(nm, place):
    """in-process geometric check against the raw file; returns (cost_weighted, cost_area) or None."""
    bins_w, items = io.read(nm, area=False)
    if sorted((p[2], p[3]) for p in place) != sorted(items):
        return None
    used = {}
    for t, k, w, h, x0, y0 in place:
        if not (0 <= t < len(bins_w)):
            return None
        W, H, _ = bins_w[t]
        if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            return None
        used.setdefault((t, k), []).append((x0, y0, w, h))
    for rs in used.values():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                xa, ya, wa, ha = rs[a]; xb, yb, wb, hb = rs[b]
                if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                    return None
    cw = sum(bins_w[t][2] for (t, k) in used)
    ca = sum(bins_w[t][0] * bins_w[t][1] for (t, k) in used)
    return cw, ca


def load_place(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    try:
        return json.load(open(p))['place']
    except Exception:
        return None


def candidates(nm, rows):
    files = set()
    for r in rows:
        if r['name'] == nm and r['sol_file'] and str(r.get('checked', '0')) == '1':
            files.add(r['sol_file'])
    for f in (os.path.join('results', 'solutions', f'{nm}_area.json'),
              os.path.join('results', 'solutions', f'{nm}_weighted.json')):
        if os.path.exists(os.path.join(ROOT, f)):
            files.add(f)
    for v in ('weighted', 'area'):
        for meth in SOL_METHODS:
            f = os.path.join('results', 'solutions', 'sprint', f'{nm}_{v}_{meth}.json')
            if os.path.exists(os.path.join(ROOT, f)):
                files.add(f)
    out = []
    for f in sorted(files):
        pl = load_place(f)
        if not pl:
            continue
        c = verify_place(nm, pl)
        if c is None:
            print(f'  WARN: packing {f} failed in-process verification; ignored', flush=True)
            continue
        out.append(dict(file=f, place=pl, weighted=c[0], area=c[1]))
    return out


def check(nm, rel, variant):
    """run the standalone checker; return its cost or None."""
    try:
        r = subprocess.run([PY, os.path.join(ROOT, 'src', 'check_solution.py'), io.instance_path(nm),
                            os.path.join(ROOT, rel), variant], capture_output=True, text=True,
                           timeout=60, cwd=ROOT)
    except Exception:
        return None
    o = r.stdout.strip()
    if r.returncode == 0 and o.startswith('valid') and ' cost ' in o:
        return int(o.rsplit(' cost ', 1)[1])
    return None


def store(nm, variant, meth, place, cost, extra):
    """write the packing (one file per cost); returns (rel_path, cost, checked)."""
    os.makedirs(SOL_DIR, exist_ok=True)
    rel = ledger.sol_relpath(nm, variant, meth, cost)
    absf = os.path.join(ROOT, rel)
    bins, _ = io.read(nm, area=(variant == 'area'))
    d = dict(name=nm, variant=variant, method=meth, cost=cost, bins=bins,
             place=[list(map(int, p)) for p in place])
    d.update(extra)
    tmp = absf + f'.tmp{os.getpid()}'
    json.dump(d, open(tmp, 'w'))
    os.replace(tmp, absf)
    cc = check(nm, rel, variant)
    return rel, cost, int(cc == cost)


def is_closed(b):
    return b is not None and b['lb'] is not None and b['ub'] is not None and b['lb'] >= b['ub'] \
        and b['ub_checked'] == 1


def run(names, variant, tl, workers, tag, log_csv, model='rep', deadline=None):
    if model == 'rep':
        import sweep_rep as sweep_model  # deferred: slow ortools import
    else:
        import sweep_model
    for nm in names:
        if deadline is not None and time.time() + tl + 5 > deadline:
            print(f'deadline reached before {nm}; stopping', flush=True)
            break
        rows = ledger.rows()
        best = ledger.best()
        b = best.get((nm, variant))
        if is_closed(b):
            print(f'{nm} {variant} skip closed lb=ub={b["ub"]} ({b["lb_method"]}/{b["ub_method"]})', flush=True)
            continue
        cands = candidates(nm, rows)
        csv_ub = io.bk[nm][0]
        hint = min(cands, key=lambda c: c[variant]) if cands else None
        ub = csv_ub if hint is None else min(csv_ub, hint[variant])
        if hint is not None and hint[variant] > csv_ub and io.n_items(nm) >= 60:
            # n >= 60: a from-scratch search for a packing <= CSV BKS rarely succeeds in 30 s; use the
            # cheapest CHECKED packing as UB so that its hint is feasible (LB validity unchanged)
            ub = hint[variant]
        stop_at = b['lb'] if (b is not None and b['lb'] is not None) else None
        t0 = time.time()
        r = sweep_model.solve(nm, variant, tl, workers=workers, ub=ub,
                              hint_place=hint['place'] if hint else None, stop_at=stop_at)
        wall = time.time() - t0
        st = r['status']
        rel = ''; checked = 0; ubrec = None
        if r['val'] is not None:
            rel, ubrec, checked = store(nm, variant, METHOD, r['place'], r['val'],
                                        dict(status=st, lb=r['lb'], time_s=round(r['time'], 1), threads=workers))
        lb = r['lb']
        if st == 'OPTIMAL' and lb != r['val']:
            lb = r['lb_raw']
        status = st
        if st == 'OPTIMAL' and not checked:
            status = 'OPTIMAL_UNCHECKED'
        note = (f'tl={tl} ub_in={ub} csv_ub={csv_ub} hint={"%s:%s" % (hint["file"].split("/")[-1], hint[variant]) if hint else "none"}'
                f' hinted={int(r.get("hinted", False))} stop_at={stop_at} lb_raw={r["lb_raw"]} kept={len(r.get("kept", []))} tag={tag}')
        if st == 'INFEASIBLE':
            note += ' WARNING_INFEASIBLE_given_ub'
            lb = None
        ledger.record(nm, variant, METHOD, status, lb=lb, ub=ubrec, time_s=wall, threads=workers,
                      sol_file=rel, checked=checked, note=note)
        # cross-variant UB
        xinfo = ''
        if r['val'] is not None and checked:
            ov = other(variant)
            c = verify_place(nm, r['place'])
            if c is not None:
                oc = c[0] if ov == 'weighted' else c[1]
                ob = best.get((nm, ov))
                worth = (ob is None or ob['ub'] is None or oc < ob['ub'])
                if ov == 'area':
                    worth = worth and oc <= io.bk[nm][0]  # only if it matches / beats the CSV BKS
                if worth:
                    rel2, oc2, ck2 = store(nm, ov, 'sweep_cross', r['place'], oc, dict(source=rel))
                    ledger.record(nm, ov, 'sweep_cross', 'UB_ONLY', ub=oc2, time_s=0, threads=0,
                                  sol_file=rel2, checked=ck2, note=f'cross-variant re-costing of {rel}')
                    xinfo = f' cross_{ov}={oc2}'
        line = (f'{nm} {variant} {st} val={r["val"]} lb={lb} csv={csv_ub}/{io.bk[nm][1]} ub_in={ub} '
                f'hint={hint[variant] if hint else None} checked={checked} t={wall:.1f}{xinfo}')
        print(line, flush=True)
        with open(log_csv, 'a') as f:
            f.write(','.join(map(str, [time.strftime('%H:%M:%S'), nm, io.n_items(nm), variant, st, r['val'], lb,
                                       r['lb_raw'], csv_ub, io.bk[nm][1], ub, hint[variant] if hint else '',
                                       checked, round(wall, 1), workers, tl, tag])) + '\n')


def group_names(n, classes=None):
    j0 = n // 20 * 10 - 9
    cl = classes or range(1, 11)
    return [f'MB_C{c}_{j}' for c in cl for j in range(j0, j0 + 10)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['weighted', 'area'], required=True)
    ap.add_argument('--time', type=float, default=20.0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--group', type=int, default=None, help='n = 20/40/60/80/100')
    ap.add_argument('--classes', default=None, help='comma list of classes')
    ap.add_argument('--shard', default='0/1', help='i/k: take names[i::k]')
    ap.add_argument('--tag', default='')
    ap.add_argument('--model', choices=['rep', 'direct'], default='rep')
    ap.add_argument('--deadline', default=None, help='HH:MM local; no solve is started that could end after it')
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    cls = [int(c) for c in a.classes.split(',')] if a.classes else None
    names = list(a.names)
    if a.group:
        names += group_names(a.group, cls)
    i, k = map(int, a.shard.split('/'))
    names = names[i::k]
    os.makedirs(OUT_DIR, exist_ok=True)
    log_csv = os.path.join(OUT_DIR, 'sweep_runs.csv')
    METHOD = 'sweep_rep' if a.model == 'rep' else 'sweep_direct'
    dl = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(':'))
        lt = time.localtime()
        dl = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        if dl < time.time() - 12 * 3600:
            dl += 86400
    run(names, a.variant, a.time, a.workers, a.tag or f'{a.variant}_{a.group}_{a.shard}', log_csv, a.model, dl)
